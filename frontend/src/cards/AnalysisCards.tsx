import { useState } from 'react';
import { CheckCircle, XCircle } from 'lucide-react';
import { api } from '../api';
import type { PharmState } from '../types';
import { Zoomable } from './shared';
import { AXIS_TITLE, errorText, fmt, niceTicks } from './helpers';

function BeCard({ r }: { r: PharmState['be_results'] }) {
  if (!r) return null;
  if (r.status !== 'ok') {
    return <div className="qc-card conditional"><div className="qc-title">Bioequivalence — not run</div>
      <div style={{ fontSize: 12 }}>{r.message}</div></div>;
  }
  const be = r.bioequivalent;
  return (
    <div className={`qc-card ${be ? 'pass' : 'fail'}`}>
      <div className="qc-title">
        {be ? <CheckCircle size={14} style={{ display: 'inline', marginRight: 6 }} />
            : <XCircle size={14} style={{ display: 'inline', marginRight: 6 }} />}
        {be ? 'Bioequivalent' : 'Not bioequivalent'} — {r.test_level} vs {r.reference_level}
      </div>
      <div style={{ fontSize: 12, color: 'var(--text-dim)', marginBottom: 6 }}>
        {r.design} · limits {r.limits?.[0]}–{r.limits?.[1]}% · n {r.n_test}/{r.n_reference}
      </div>
      <table className="nca-table">
        <thead><tr><th>Parameter</th><th>GMR %</th><th>90% CI</th><th>intra-CV%</th><th></th></tr></thead>
        <tbody>
          {Object.entries(r.parameters ?? {}).map(([p, v]) => (
            <tr key={p}>
              <td>{p}</td>
              <td>{fmt(v.gmr_pct ?? undefined, 1)}</td>
              <td>{fmt(v.ci_lower_pct ?? undefined, 1)}–{fmt(v.ci_upper_pct ?? undefined, 1)}</td>
              <td>{v.cv_intra_pct == null ? '–' : fmt(v.cv_intra_pct, 1)}</td>
              <td style={{ color: v.within_limits ? 'var(--green)' : 'var(--red)' }}>
                {v.within_limits ? '✓' : '✗'}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function StatsAdviceCard({ r }: { r: PharmState['stats_advice'] }) {
  if (!r || r.status !== 'ok') return null;
  const d = r.design;
  const nonPar = Object.values(r.metrics).filter(m => m.family === 'non-parametric' && m.scale === 'rank').length;
  const fmtP = (p: number | null | undefined) => (p == null ? '–' : p < 0.001 ? '<0.001' : p.toFixed(3));
  return (
    <div className={`qc-card ${nonPar > 1 ? 'conditional' : 'pass'}`}>
      <div className="qc-title">
        Statistical analysis plan — {d.design_label}, {d.n_groups} group{d.n_groups === 1 ? '' : 's'}
        {d.group_var ? ` by ${d.group_var}` : ''}, {d.n_subjects} subjects
      </div>
      <div style={{ fontSize: 12, color: 'var(--text-dim)', marginBottom: 6 }}>
        exposures from {d.source === 'nca' ? 'NCA parameters' : 'observed concentrations'}
        {d.paired ? ' · paired (same subjects in every group)' : ''}
      </div>
      <table className="nca-table">
        <thead><tr><th>Metric</th><th>n</th><th>Shapiro (log)</th><th>Scale</th><th>Family</th><th>Primary test</th></tr></thead>
        <tbody>
          {Object.entries(r.metrics).map(([m, v]) => (
            <tr key={m} title={v.rationale}>
              <td>{m}</td>
              <td>{v.n}</td>
              <td>{fmtP(v.shapiro_log_p)}</td>
              <td>{v.scale}</td>
              <td style={{ color: v.family === 'parametric' ? 'var(--green)' : 'var(--yellow)' }}>{v.family}</td>
              <td style={{ fontSize: 11 }}>{v.primary_test}{v.sensitivity_test ? ` (sensitivity: ${v.sensitivity_test})` : ''}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <ul style={{ fontSize: 12, margin: '8px 0 0', paddingLeft: 18 }}>
        {r.recommendations.map(rec => (
          <li key={rec.topic}><b>{rec.topic}:</b> {rec.recommendation} <span style={{ color: 'var(--text-dim)' }}>— {rec.rationale}</span></li>
        ))}
        {r.covariates.map(c => (
          <li key={c.name}><b>{c.name}</b> ({c.kind}): {c.recommendation}</li>
        ))}
      </ul>
      {r.caveats.length > 0 && (
        <div style={{ fontSize: 12, color: 'var(--text)', marginTop: 6 }}>
          {r.caveats.map((c, i) => <div key={i}>⚠ {c}</div>)}
        </div>
      )}
    </div>
  );
}

function DosePropCard({ r }: { r: PharmState['dose_prop_results'] }) {
  if (!r) return null;
  if (r.status !== 'ok') {
    return <div className="qc-card conditional"><div className="qc-title">Dose proportionality — not assessed</div>
      <div style={{ fontSize: 12 }}>{r.message}</div></div>;
  }
  const prop = r.proportional;
  return (
    <div className={`qc-card ${prop ? 'pass' : 'conditional'}`}>
      <div className="qc-title">
        {prop ? 'Dose-proportional' : 'Not dose-proportional'} (power model)
      </div>
      <div style={{ fontSize: 12, color: 'var(--text-dim)', marginBottom: 6 }}>
        dose levels {(r.dose_levels ?? []).join(', ')} mg
      </div>
      <table className="nca-table">
        <thead><tr><th>Parameter</th><th>slope β</th><th>90% CI</th><th>critical region</th><th></th></tr></thead>
        <tbody>
          {Object.entries(r.parameters ?? {}).map(([p, v]) => (
            <tr key={p}>
              <td>{p}</td>
              <td>{fmt(v.slope ?? undefined, 3)}</td>
              <td>{fmt(v.slope_ci_lower ?? undefined, 2)}–{fmt(v.slope_ci_upper ?? undefined, 2)}</td>
              <td>{v.critical_region ? `${fmt(v.critical_region[0], 2)}–${fmt(v.critical_region[1], 2)}` : '–'}</td>
              <td style={{ color: v.proportional ? 'var(--green)' : 'var(--yellow)' }}>
                {v.proportional ? '✓' : '!'}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function CompartmentalCard({ r }: { r: PharmState['compartmental_results'] }) {
  if (!r) return null;
  const counts = r.model_selection_counts ?? {};
  const ss = r.steady_state === true;
  const n1 = (counts['1cmt'] ?? 0) + (counts['1cmt_ss'] ?? 0);
  const n2 = (counts['2cmt'] ?? 0) + (counts['2cmt_ss'] ?? 0);
  const modelLabel = (m: string) =>
    ({ '1cmt': '1-cmt', '2cmt': '2-cmt', '1cmt_ss': '1-cmt SS', '2cmt_ss': '2-cmt SS' }[m] ?? m);
  return (
    <div>
      <div style={{ fontSize: 12, color: 'var(--text-dim)', marginBottom: 4 }}>
        {ss ? 'Steady-state compartmental fit' : 'Compartmental fit'} — {r.n_converged}/{r.n_subjects} converged ·
        {' '}{n1}×1-cmt, {n2}×2-cmt (by AIC)
      </div>
      <table className="nca-table">
        <thead><tr><th>ID</th><th>Model</th><th>ka</th><th>CL/F</th><th>V/F</th><th>AIC</th><th>R²</th></tr></thead>
        <tbody>
          {r.fits.map(f => {
            const p = f.params ?? {};
            return (
              <tr key={String(f.subject)}>
                <td>{f.subject}</td>
                <td>{modelLabel(f.model)}{!f.converged && ' ✗'}</td>
                <td>{fmt(p.ka, 2)}</td>
                <td>{fmt(p.CL, 2)}</td>
                <td>{fmt(p.V ?? p.V1, 1)}</td>
                <td>{fmt(f.aic ?? undefined, 1)}</td>
                <td>{fmt(f.r_squared ?? undefined, 3)}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

function PopPkCard({ r }: { r: PharmState['poppk_results'] }) {
  if (!r) return null;
  if (r.status !== 'ok') {
    return <div className="qc-card conditional"><div className="qc-title">Population PK — not run</div>
      <div style={{ fontSize: 12 }}>{r.message}</div></div>;
  }
  const cov = r.covariate_screen as { covariate?: string; pearson_r?: number; slope?: number } | undefined;
  return (
    <div>
      <div style={{ fontSize: 12, color: 'var(--text-dim)', marginBottom: 4 }}>
        Population PK — two-stage ({r.source}) · {r.n_subjects} subjects
      </div>
      <table className="nca-table">
        <thead><tr><th>Parameter</th><th>typical (GM)</th><th>IIV CV%</th><th>median</th><th>n</th></tr></thead>
        <tbody>
          {Object.entries(r.parameters ?? {}).map(([p, v]) => (
            <tr key={p}>
              <td>{p}</td>
              <td>{fmt(v.typical_value ?? undefined, 2)}</td>
              <td>{v.iiv_cv_pct == null ? '–' : fmt(v.iiv_cv_pct, 1)}</td>
              <td>{fmt(v.median ?? undefined, 2)}</td>
              <td>{v.n}</td>
            </tr>
          ))}
        </tbody>
      </table>
      {cov?.covariate && (
        <div style={{ fontSize: 11, color: 'var(--text-dim)', marginTop: 6 }}>
          covariate {cov.covariate} on CL/F: r = {fmt(cov.pearson_r, 2)}, slope = {fmt(cov.slope, 3)}
        </div>
      )}
    </div>
  );
}

// ── Clinical-pharmacology calculators (PharmKit toolkit inside PharmAgent) ──
type CalcField = { name: string; label: string; unit?: string; default?: string; optional?: boolean;
  options?: string[] };
type CalcForm = { tool: string; label: string; fields: CalcField[]; hint: string };
const CALC_FORMS: CalcForm[] = [
  { tool: 'calc_half_life', label: 'Half-life & rate constant',
    hint: 'Give a half-life OR ke, OR two terminal points (c1@t1, c2@t2).',
    fields: [{ name: 'half_life', label: 't½', unit: 'h', optional: true }, { name: 'ke', label: 'ke', unit: '1/h', optional: true },
      { name: 'c1', label: 'C1', optional: true }, { name: 't1', label: 't1', unit: 'h', optional: true },
      { name: 'c2', label: 'C2', optional: true }, { name: 't2', label: 't2', unit: 'h', optional: true }] },
  { tool: 'calc_accumulation', label: 'Accumulation & steady state',
    hint: 'Rac = 1/(1 − e^(−ke·τ)); time to a fraction of steady state.',
    fields: [{ name: 'tau', label: 'τ', unit: 'h', default: '24' }, { name: 'half_life', label: 't½', unit: 'h', optional: true },
      { name: 'ke', label: 'ke', unit: '1/h', optional: true }, { name: 'fraction', label: 'fraction of SS', default: '0.9', optional: true }] },
  { tool: 'calc_dose_regimen', label: 'Loading & maintenance dose',
    hint: 'LD = C_target·V/F; MD = C_avg,ss·CL·τ/F. Fill either or both parts.',
    fields: [{ name: 'target_conc', label: 'C target', unit: 'mg/L', optional: true }, { name: 'volume', label: 'V', unit: 'L', optional: true },
      { name: 'cavg_ss', label: 'C avg,ss', unit: 'mg/L', optional: true }, { name: 'clearance', label: 'CL', unit: 'L/h', optional: true },
      { name: 'tau', label: 'τ', unit: 'h', optional: true }, { name: 'bioavailability', label: 'F', default: '1', optional: true }] },
  { tool: 'calc_renal_function', label: 'Renal function & dose',
    hint: 'Cockcroft-Gault (needs weight) + CKD-EPI 2021; optional clearance-proportional dose adjustment.',
    fields: [{ name: 'age', label: 'age', unit: 'y' }, { name: 'scr_mg_dl', label: 'SCr', unit: 'mg/dL' },
      { name: 'sex', label: 'sex', options: ['male', 'female'] }, { name: 'weight_kg', label: 'weight', unit: 'kg', optional: true },
      { name: 'normal_dose', label: 'normal dose', optional: true }, { name: 'reference_crcl', label: 'ref CrCl', unit: 'mL/min', default: '120', optional: true },
      { name: 'fraction_renal', label: 'fraction renal', default: '1', optional: true }] },
  { tool: 'calc_allometric', label: 'Allometric scaling',
    hint: 'Y2 = Y1·(BW2/BW1)^exponent — 0.75 for CL, 1.0 for V by default.',
    fields: [{ name: 'value', label: 'value' }, { name: 'from_bw', label: 'from BW', unit: 'kg', default: '70' },
      { name: 'to_bw', label: 'to BW', unit: 'kg' }, { name: 'kind', label: 'kind', options: ['CL', 'V'] },
      { name: 'exponent', label: 'exponent', optional: true }] },
  { tool: 'convert_concentration', label: 'mg/L ↔ µmol/L',
    hint: 'µM = mg/L / MW · 1000. Fill one side.',
    fields: [{ name: 'molar_mass', label: 'MW', unit: 'g/mol' }, { name: 'mg_per_l', label: 'mg/L', optional: true },
      { name: 'umol_per_l', label: 'µmol/L', optional: true }] },
  { tool: 'calc_be_sample_size', label: 'BE sample size (2×2)',
    hint: 'TOST normal approximation (Chow & Liu); CV as fraction or %.',
    fields: [{ name: 'cv_intra', label: 'intra CV', default: '0.25' }, { name: 'gmr', label: 'GMR', default: '0.95', optional: true },
      { name: 'power', label: 'power', default: '0.8', optional: true }, { name: 'alpha', label: 'α', default: '0.05', optional: true },
      { name: 'lower', label: 'lower', default: '0.8', optional: true }, { name: 'upper', label: 'upper', default: '1.25', optional: true }] },
  { tool: 'quick_one_compartment', label: 'One-compartment profile',
    hint: 'Analytic 1-cmt profile; ke, t½ or CL fixes elimination.',
    fields: [{ name: 'route', label: 'route', options: ['oral', 'iv', 'infusion'] }, { name: 'dose', label: 'dose', unit: 'mg', default: '100' },
      { name: 'volume', label: 'V', unit: 'L', default: '50' }, { name: 'half_life', label: 't½', unit: 'h', optional: true },
      { name: 'ke', label: 'ke', unit: '1/h', optional: true }, { name: 'clearance', label: 'CL', unit: 'L/h', optional: true },
      { name: 'ka', label: 'ka', unit: '1/h', default: '1', optional: true }, { name: 'bioavailability', label: 'F', default: '1', optional: true },
      { name: 't_inf', label: 't inf', unit: 'h', optional: true }, { name: 't_end', label: 't end', unit: 'h', optional: true }] },
];
const CALC_OUTPUT_LABEL: Record<string, string> = {
  ke: 'ke', half_life: 't½', time_to_90pct_ss: 't to 90% SS', time_to_97pct_ss: 't to 97% SS',
  accumulation_ratio: 'Rac', time_to_fraction_ss: 't to fraction SS', doses_to_fraction_ss: 'doses to fraction SS', fraction: 'fraction',
  loading_dose: 'loading dose', maintenance_dose: 'maintenance dose / τ', dose_rate: 'dose rate (per h)',
  egfr_ckd_epi_2021: 'eGFR CKD-EPI 2021 (mL/min/1.73m²)', crcl_cockcroft_gault: 'CrCl Cockcroft-Gault (mL/min)',
  adjusted_dose: 'adjusted dose', adjustment_basis: 'adjustment basis', scaled_value: 'scaled value', ratio: 'ratio',
  umol_per_l: 'µmol/L', mg_per_l: 'mg/L', total: 'total N', per_sequence: 'per sequence', sigma_w: 'σ_w (log)',
  c0: 'C0', auc_inf: 'AUC∞', c_end_of_infusion: 'C end of infusion', tmax: 'tmax', cmax: 'Cmax',
};

function ClinpharmCard({ r }: { r: PharmState['clinpharm_results'] }) {
  if (!r || r.status !== 'ok') return null;
  const fmtv = (v: unknown) => typeof v === 'number' ? (Number.isInteger(v) ? String(v) : fmt(v, Math.abs(v) >= 100 ? 1 : 4)) : String(v ?? '–');
  const profile = Array.isArray(r.outputs.profile) ? r.outputs.profile as { t: number; c: number }[] : null;
  const entries = Object.entries(r.outputs).filter(([k]) => k !== 'profile');
  return (
    <div>
      <div style={{ fontSize: 12, color: 'var(--text-dim)', marginBottom: 4 }}>{r.label} · {r.formula}</div>
      <div style={{ display: 'flex', gap: 16, flexWrap: 'wrap' }}>
        <table className="nca-table" style={{ width: 'auto' }}>
          <thead><tr><th>Input</th><th>Value</th></tr></thead>
          <tbody>{Object.entries(r.inputs).map(([k, v]) => <tr key={k}><td>{k}</td><td>{fmtv(v)}</td></tr>)}</tbody>
        </table>
        <table className="nca-table" style={{ width: 'auto' }}>
          <thead><tr><th>Result</th><th>Value</th></tr></thead>
          <tbody>{entries.map(([k, v]) => (
            <tr key={k}><td>{CALC_OUTPUT_LABEL[k] ?? k}</td><td><b>{fmtv(v)}</b></td></tr>
          ))}</tbody>
        </table>
      </div>
      {profile && profile.length > 1 && <OneCompProfile pts={profile} />}
      {r.note && <div style={{ fontSize: 11, color: 'var(--yellow)', marginTop: 6 }}>{r.note}</div>}
    </div>
  );
}

function OneCompProfile({ pts }: { pts: { t: number; c: number }[] }) {
  const W = 420, H = 170, ml = 48, mr = 10, mt = 10, mb = 32;
  const tmax = pts[pts.length - 1].t || 1, cmax = Math.max(...pts.map(p => p.c)) * 1.05 || 1;
  const sx = (t: number) => ml + (t / tmax) * (W - ml - mr);
  const sy = (c: number) => H - mb - (c / cmax) * (H - mt - mb);
  const d = pts.map((p, i) => `${i ? 'L' : 'M'}${sx(p.t).toFixed(1)},${sy(p.c).toFixed(1)}`).join(' ');
  return (
    <Zoomable title="One-compartment profile">
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: '100%', maxWidth: W, marginTop: 8 }} role="img" aria-label="One-compartment profile">
      <line x1={ml} y1={H - mb} x2={W - mr} y2={H - mb} stroke="var(--border)" />
      <line x1={ml} y1={mt} x2={ml} y2={H - mb} stroke="var(--border)" />
      {niceTicks(0, tmax, 5).map(v => <text key={`x${v}`} x={sx(v)} y={H - mb + 12} textAnchor="middle" fontSize="9" fill="var(--text-dim)">{v}</text>)}
      {niceTicks(0, cmax, 4).map(v => <text key={`y${v}`} x={ml - 5} y={sy(v) + 3} textAnchor="end" fontSize="9" fill="var(--text-dim)">{fmt(v, 2)}</text>)}
      <path d={d} fill="none" stroke="var(--accent)" strokeWidth={1.6} />
      <text x={(ml + W - mr) / 2} y={H - 6} textAnchor="middle" {...AXIS_TITLE}>time (h)</text>
      <text x={13} y={(mt + H - mb) / 2} textAnchor="middle" {...AXIS_TITLE} transform={`rotate(-90 13 ${(mt + H - mb) / 2})`}>concentration</text>
    </svg>
    </Zoomable>
  );
}

/** Calculator panel: pick a tool, fill its inputs, run through /calc (audited, state-written). */
function CalculatorsPanel({ sessionId, busy, onResult }: {
  sessionId: string; busy: boolean; onResult: (state: PharmState, summary: string) => void;
}) {
  const [idx, setIdx] = useState(0);
  const [vals, setVals] = useState<Record<string, string>>({});
  const [err, setErr] = useState('');
  const [running, setRunning] = useState(false);
  const form = CALC_FORMS[idx];
  const pick = (i: number) => { setIdx(i); setVals({}); setErr(''); };
  const run = async () => {
    setRunning(true); setErr('');
    const args: Record<string, string> = {};
    for (const f of form.fields) {
      const v = (vals[f.name] ?? f.default ?? (f.options ? f.options[0] : '')).trim();
      if (v !== '') args[f.name] = v;
    }
    try {
      const res = await api.calc(sessionId, form.tool, args);
      onResult(res.state, res.summary);
    } catch (e) { setErr(errorText(e)); } finally { setRunning(false); }
  };
  const field = { fontSize: 12, padding: '3px 6px', borderRadius: 6, border: '1px solid var(--border)', background: 'var(--bg)', color: 'var(--text)', width: 90 } as const;
  return (
    <div style={{ border: '1px solid var(--border)', borderRadius: 10, padding: '8px 10px', marginTop: 6, background: 'var(--bg-panel)' }}>
      <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap', marginBottom: 6 }}>
        {CALC_FORMS.map((f, i) => (
          <button key={f.tool} type="button" className="chip" aria-pressed={i === idx} onClick={() => pick(i)}>{f.label}</button>
        ))}
      </div>
      <div style={{ fontSize: 11, color: 'var(--text-dim)', marginBottom: 6 }}>{form.hint}</div>
      <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', alignItems: 'flex-end' }}>
        {form.fields.map(f => (
          <label key={f.name} style={{ fontSize: 11, color: 'var(--text-dim)', display: 'flex', flexDirection: 'column', gap: 2 }}>
            <span>{f.label}{f.unit ? ` (${f.unit})` : ''}{f.optional ? '' : ' *'}</span>
            {f.options ? (
              <select style={field} value={vals[f.name] ?? f.options[0]} onChange={e => setVals({ ...vals, [f.name]: e.target.value })}>
                {f.options.map(o => <option key={o} value={o}>{o}</option>)}
              </select>
            ) : (
              <input style={field} inputMode="decimal" placeholder={f.default ?? ''} value={vals[f.name] ?? ''}
                onChange={e => setVals({ ...vals, [f.name]: e.target.value })} />
            )}
          </label>
        ))}
        <button className="btn btn-green" disabled={busy || running} onClick={run}>{running ? 'Running…' : 'Calculate'}</button>
      </div>
      {err && <div style={{ fontSize: 11, color: 'var(--red)', marginTop: 6 }}>{err}</div>}
    </div>
  );
}

export {
  BeCard,
  StatsAdviceCard,
  DosePropCard,
  CompartmentalCard,
  PopPkCard,
  ClinpharmCard,
  OneCompProfile,
  CalculatorsPanel,
};
