import { useState } from 'react';
import type { PcVpcBin, PharmState } from '../types';
import { Zoomable } from './shared';
import { fmt, fmtIIV } from './helpers';

function PkPopTable({ pop }: { pop: { parameters: Record<string, { typical_value: number | null; iiv_cv_pct: number | null; median: number | null; n: number }> } }) {
  const entries = Object.entries(pop.parameters ?? {});
  if (entries.length === 0) return null;
  const anyUnstable = entries.some(([, v]) => (v.iiv_cv_pct ?? 0) > 300);
  return (
    <>
      <table className="nca-table">
        <thead><tr><th>Parameter</th><th>typical (GM)</th><th>IIV CV%</th><th>median</th><th>n</th></tr></thead>
        <tbody>
          {entries.map(([k, v]) => {
            const iiv = fmtIIV(v.iiv_cv_pct);
            return (
              <tr key={k}>
                <td>{k}</td>
                <td>{fmt(v.typical_value ?? undefined, 2)}</td>
                <td style={iiv.unstable ? { color: 'var(--yellow)' } : {}}>{iiv.text}</td>
                <td>{fmt(v.median ?? undefined, 2)}</td>
                <td>{v.n}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
      {anyUnstable && (
        <div style={{ fontSize: 11, color: 'var(--yellow)', marginTop: 4 }}>
          ⚠ Very high IIV signals an over-parameterized model (unstable individual estimates) — prefer a simpler model.
        </div>
      )}
    </>
  );
}

function PkModelCard({ r }: { r: PharmState['pk_model_results'] }) {
  if (!r) return null;
  if (r.status !== 'ok') {
    return <div className="qc-card conditional"><div className="qc-title">PK model — not run</div>
      <div style={{ fontSize: 12 }}>{r.message}</div></div>;
  }
  if (r.mode === 'compare' && r.ranking) {
    return (
      <div>
        <div style={{ fontSize: 12, color: 'var(--text-dim)', marginBottom: 4 }}>
          Model comparison — {r.ranking.length} models · best by AIC:
          {' '}<span style={{ color: 'var(--green)' }}>{r.best?.label ?? r.best_model}</span>
          {r.multiple_dose && ' · multiple-dose'}
        </div>
        <table className="nca-table">
          <thead><tr><th>Model</th><th>Converged</th><th>Total AIC</th><th>Mean AIC</th></tr></thead>
          <tbody>
            {r.ranking.map((row, i) => (
              <tr key={row.model_key} style={i === 0 ? { color: 'var(--green)' } : {}}>
                <td>{row.label}{i === 0 && ' ✓'}</td>
                <td>{row.n_converged}/{row.n_subjects}</td>
                <td>{fmt(row.total_aic ?? undefined, 1)}</td>
                <td>{fmt(row.mean_aic ?? undefined, 1)}</td>
              </tr>
            ))}
          </tbody>
        </table>
        {r.best?.population && (
          <div style={{ marginTop: 8 }}>
            <div style={{ fontSize: 11, color: 'var(--text-dim)', marginBottom: 4 }}>
              {r.best.label} — population (two-stage typical · IIV)
            </div>
            <PkPopTable pop={r.best.population} />
          </div>
        )}
      </div>
    );
  }
  // fit mode
  const fits = r.individual_fits ?? [];
  const paramKeys = fits.find(f => f.params)?.params ? Object.keys(fits.find(f => f.params)!.params!) : [];
  return (
    <div>
      <div style={{ fontSize: 12, color: 'var(--text-dim)', marginBottom: 4 }}>
        {r.label} — {r.n_converged}/{r.n_subjects} converged · mean AIC {fmt(r.mean_aic ?? undefined, 1)}
        {r.is_pkpd && ' · PK/PD dual-endpoint'}
        {r.multiple_dose && ' · multiple-dose'}
      </div>
      {r.population && <PkPopTable pop={r.population} />}
      {fits.length > 0 && paramKeys.length > 0 && (
        <div style={{ marginTop: 8 }}>
          <div style={{ fontSize: 11, color: 'var(--text-dim)', marginBottom: 4 }}>Per-subject estimates</div>
          <table className="nca-table">
            <thead><tr><th>ID</th>{paramKeys.map(k => <th key={k}>{k}</th>)}<th>AIC</th><th>R²</th></tr></thead>
            <tbody>
              {fits.map(f => (
                <tr key={String(f.subject)}>
                  <td>{f.subject}</td>
                  {paramKeys.map(k => <td key={k}>{fmt(f.params?.[k], 2)}</td>)}
                  <td>{fmt(f.aic ?? undefined, 1)}</td>
                  <td>{fmt(f.r_squared ?? undefined, 3)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

/** Bayesian-borrowing diagnostics for a MAP fit: a prior-predictive band vs the
 * observed data + a prior-vs-posterior shrinkage table (Week-15). */
function PriorCheckCard({ r }: { r: PharmState['prior_check_results'] }) {
  if (!r || r.status !== 'ok') {
    return <div className="qc-card conditional"><div className="qc-title">Prior check — not run</div>
      <div style={{ fontSize: 12 }}>{r?.message}</div></div>;
  }
  const ppc = r.prior_predictive;
  const rows = r.diagnostic?.params ?? [];
  const band = (ppc?.band ?? []).filter(b => b.time != null && b.lo != null && b.hi != null);
  const obs = ppc?.observed ?? [];
  // prior-predictive band SVG (log-y)
  const W = 460, H = 220, ml = 46, mr = 10, mt = 10, mb = 30;
  const ys = band.flatMap(b => [b.lo, b.hi]).concat(obs.map(o => o.dv)).filter(v => v != null && (v as number) > 0) as number[];
  const xs = band.map(b => b.time as number).concat(obs.map(o => o.time as number)).filter(v => v != null);
  const hasBand = band.length > 1 && ys.length > 0;
  const lo = hasBand ? Math.max(1e-6, Math.min(...ys) * 0.8) : 0.1;
  const hi = hasBand ? Math.max(...ys) * 1.2 : 1;
  const xmax = xs.length ? Math.max(...xs) : 1;
  const lnLo = Math.log(lo), lnHi = Math.log(hi);
  const sx = (t: number) => ml + (xmax > 0 ? t / xmax : 0) * (W - ml - mr);
  const sy = (v: number) => H - mb - ((Math.log(Math.max(v, 1e-6)) - lnLo) / (lnHi - lnLo || 1)) * (H - mt - mb);
  const areaPts = hasBand
    ? band.map(b => `${sx(b.time as number)},${sy(b.hi as number)}`).join(' ') + ' ' +
      band.slice().reverse().map(b => `${sx(b.time as number)},${sy(b.lo as number)}`).join(' ')
    : '';
  const medPts = hasBand ? band.filter(b => b.med != null).map(b => `${sx(b.time as number)},${sy(b.med as number)}`).join(' ') : '';
  return (
    <div>
      <div style={{ fontSize: 12, color: 'var(--text-dim)', marginBottom: 6 }}>
        {r.label} · Bayesian borrowing · mean shrinkage <b>{fmt(r.diagnostic?.mean_shrinkage ?? undefined, 2)}</b>
        {ppc?.coverage_pct != null && <> · prior-predictive coverage <b>{fmt(ppc.coverage_pct, 0)}%</b> ({ppc.n_draws} draws)</>}
      </div>
      {hasBand && (
        <>
          <div style={{ fontSize: 11, color: 'var(--text-dim)', fontWeight: 600 }}>Prior-predictive band vs observed data</div>
          <Zoomable title="VPC panel">
          <svg viewBox={`0 0 ${W} ${H}`} style={{ width: '100%', maxWidth: W }} role="img"
            aria-label="Prior-predictive concentration band vs observed data">
            <polygon points={areaPts} fill="var(--accent)" fillOpacity="0.15" />
            {medPts && <polyline points={medPts} fill="none" stroke="var(--accent)" strokeWidth="1.4" strokeDasharray="4 3" />}
            {obs.map((o, i) => (o.time != null && o.dv != null && (o.dv as number) > 0
              ? <circle key={i} cx={sx(o.time)} cy={sy(o.dv)} r="2" fill="var(--text)" fillOpacity="0.7" /> : null))}
            <line x1={ml} y1={H - mb} x2={W - mr} y2={H - mb} stroke="var(--border)" />
            <text x={(ml + W) / 2} y={H - 4} textAnchor="middle" fontSize="9" fill="var(--text-dim)">Time</text>
          </svg>
          </Zoomable>
        </>
      )}
      <table className="nca-table" style={{ marginTop: 8 }}>
        <thead><tr><th>Param</th><th>Prior mean</th><th>Posterior</th><th>95% CrI</th><th>Shrinkage</th></tr></thead>
        <tbody>
          {rows.map((p, i) => (
            <tr key={i}>
              <td>{p.param}</td>
              <td style={{ color: 'var(--text-dim)' }}>{fmt(p.prior_mean ?? undefined, 3)}</td>
              <td>{fmt(p.post_mean ?? undefined, 3)}</td>
              <td style={{ fontSize: 11, color: 'var(--text-dim)' }}>
                {p.ci95?.[0] != null ? `${fmt(p.ci95[0], 3)}–${fmt(p.ci95[1] ?? undefined, 3)}` : '—'}</td>
              <td style={{ color: p.shrinkage != null ? 'var(--green)' : 'var(--text-dim)' }}>
                {p.shrinkage != null ? fmt(p.shrinkage, 2) : '—'}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <div style={{ fontSize: 11, color: 'var(--text-dim)', marginTop: 2 }}>
        Shrinkage = 1 − posterior_sd/prior_sd (0 = data adds nothing beyond the prior; →1 = data dominates).
      </div>
    </div>
  );
}

function NlmeCard({ r }: { r: PharmState['nlme_results'] }) {
  if (!r) return null;
  if (r.status !== 'ok') {
    return <div className="qc-card conditional"><div className="qc-title">NLME — not run</div>
      <div style={{ fontSize: 12 }}>{r.message}</div></div>;
  }
  const theta = r.theta ?? {};
  const omega = r.omega_cv_pct ?? {};
  const rse = r.theta_rse_pct ?? {};
  const ormse = r.omega_rse_pct ?? {};
  const srse = r.sigma_rse_pct ?? { prop: null, add: null };
  const shr = r.shrinkage_pct ?? {};
  const iiv = new Set(r.iiv_params ?? []);
  const sig = r.sigma ?? { prop: null, add: null };
  const cond = r.condition_number;
  const condFlag = cond != null && cond > 1000;
  const sigPart = (v: number | null, rseV: number | null, label: string) =>
    v == null ? '' : `${label} ${fmt(v, 3)}${rseV != null ? ` (${fmt(rseV, 1)}% RSE)` : ''}`;
  return (
    <div>
      <div style={{ fontSize: 12, color: 'var(--text-dim)', marginBottom: 4 }}>
        {r.method} · {r.label} · OFV {fmt(r.ofv ?? undefined, 1)} · {r.n_subjects} subjects ·
        {' '}IIV on {(r.iiv_params ?? []).join(', ')} · {r.error_model} error
        {r.n_blq ? ` · ${r.n_blq} BLQ (M3)` : ''}
        {r.map && <span style={{ color: 'var(--accent)' }}> · MAP (informative prior{r.ofv_likelihood != null
          ? `, −2LL ${fmt(r.ofv_likelihood, 1)}` : ''})</span>}
        {' '}· {r.converged ? 'converged' : 'did not converge'}
        {cond != null && (
          <> · <span style={{ color: condFlag ? 'var(--red)' : 'inherit' }}>
            cond {fmt(cond, 1)}{condFlag ? ' ⚠' : ''}
          </span></>
        )}
      </div>
      {r.auto && (
        <div style={{ fontSize: 11, color: 'var(--text-dim)', marginBottom: 6 }}
          title="Every candidate is a converged FOCE-I fit of the same model on the same data, so their OFVs are directly comparable and the lowest wins.">
          {r.auto.escalated
            ? `Auto: escalated (${r.auto.reason}) — ${r.auto.n_candidates} starts compared, kept ${r.auto.winner}`
            : `Auto: no escalation (${r.auto.reason}) — kept ${r.auto.winner}`}
          {r.auto.escalated && (
            <> · OFV {Object.entries(r.auto.candidate_ofv)
              .filter(([, v]) => v != null)
              .sort((a, b) => (a[1] as number) - (b[1] as number))
              .map(([k, v]) => `${k} ${fmt(v as number, 1)}`)
              .join(' | ')}</>
          )}
        </div>
      )}
      <table className="nca-table">
        <thead><tr><th>Parameter</th><th>Typical (θ)</th><th>RSE%</th><th>IIV CV% (RSE%)</th><th>η-shrinkage%</th></tr></thead>
        <tbody>
          {Object.entries(theta).map(([p, v]) => (
            <tr key={p}>
              <td>{p}</td>
              <td>{fmt(v, 3)}</td>
              <td>{rse[p] != null ? fmt(rse[p], 1) : '–'}</td>
              <td>{iiv.has(p) && omega[p] != null
                ? `${fmt(omega[p], 1)}${ormse[p] != null ? ` (${fmt(ormse[p], 0)}%)` : ''}`
                : '–'}</td>
              <td>{iiv.has(p) && shr[p] != null ? fmt(shr[p], 1) : '–'}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <div style={{ fontSize: 11, color: 'var(--text-dim)', marginTop: 6 }}>
        Residual error — {[sigPart(sig.prop, srse.prop, 'proportional'),
          sigPart(sig.add, srse.add, 'additive')].filter(Boolean).join(' · ')}
      </div>
      {(r.covariate_effects ?? []).length > 0 && (
        <div style={{ fontSize: 11, color: 'var(--text-dim)', marginTop: 6 }}>
          Covariate effects — {(r.covariate_effects ?? []).map((ce, i) => (
            <span key={i}>{i > 0 ? ' · ' : ''}{ce.param}: {ce.description}
              {typeof ce.rse_pct === 'number' ? ` (${fmt(ce.rse_pct, 0)}% RSE)` : ''}</span>
          ))}
        </div>
      )}
      {r.cov_note ? (
        <div style={{ fontSize: 11, color: 'var(--red)', marginTop: 4 }}>
          {r.cov_note}
        </div>
      ) : null}
    </div>
  );
}

function EngineComparisonCard({ r }: { r: PharmState['engine_comparison_results'] }) {
  if (!r) return null;
  if (r.status !== 'ok') {
    return <div className="qc-card conditional"><div className="qc-title">Cross-engine comparison — not run</div>
      <div style={{ fontSize: 12 }}>{r.message}</div></div>;
  }
  const ranking = r.prediction_ranking ?? [];
  const wl = r.within_engine_likelihood ?? {};
  const skipped = (r.results ?? []).filter(x => x.status !== 'ok');
  return (
    <div>
      <div style={{ fontSize: 12, color: 'var(--text-dim)', marginBottom: 4 }}>
        {r.n_candidates} model(s) × {r.n_available}/{r.n_engines} engine(s) · winner{' '}
        <strong style={{ color: 'var(--agent-nca)' }}>{r.winner?.engine} / {r.winner?.model_name}</strong>
      </div>
      <table className="nca-table">
        <thead><tr><th>Engine</th><th>Model</th><th>pred RMSE</th><th>VPC cov90</th><th>R²</th><th>|bias|</th><th></th></tr></thead>
        <tbody>
          {ranking.map((row, i) => (
            <tr key={i} style={i === 0 ? { fontWeight: 600 } : undefined}>
              <td>{row.engine}{i === 0 ? ' ★' : ''}</td>
              <td>{row.model_name}</td>
              <td>{fmt(row.pred_rmse ?? undefined, 4)}</td>
              <td>{row.vpc_coverage90 != null ? fmt(row.vpc_coverage90, 2) : '–'}</td>
              <td>{row.pred_r2 != null ? fmt(row.pred_r2, 3) : '–'}</td>
              <td>{row.pred_bias != null ? fmt(Math.abs(row.pred_bias), 3) : '–'}</td>
              <td style={{ color: 'var(--red)' }}>{row.converged ? '' : '⚠'}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <div style={{ fontSize: 11, color: 'var(--text-dim)', marginTop: 6 }}>
        Within-engine OFV (never compared across engines):{' '}
        {Object.entries(wl).map(([eng, rows], i) => (
          <span key={eng}>{i > 0 ? ' · ' : ''}{eng}: {rows.map(x => x.ofv != null ? fmt(x.ofv, 1) : '–').join(', ')}</span>
        ))}
      </div>
      {skipped.length > 0 && (
        <div style={{ fontSize: 11, color: 'var(--text-dim)', marginTop: 4 }}>
          {skipped.map((x, i) => (
            <span key={i}>{i > 0 ? ' · ' : ''}{x.engine}: {x.status}{x.message ? ` (${x.message})` : ''}</span>
          ))}
        </div>
      )}
      <div style={{ fontSize: 11, color: 'var(--text-dim)', marginTop: 4, fontStyle: 'italic' }}>
        Ranked by prediction accuracy; OFV/AIC/BIC are not comparable across estimation algorithms.
      </div>
    </div>
  );
}

function ScmCard({ r }: { r: PharmState['scm_results'] }) {
  if (!r) return null;
  if (r.status !== 'ok') {
    return <div className="qc-card conditional"><div className="qc-title">Covariate SCM — not run</div>
      <div style={{ fontSize: 12 }}>{r.message}</div></div>;
  }
  const steps = r.steps ?? [];
  const selected = r.selected ?? [];
  const dOfv = (r.base_ofv != null && r.final_ofv != null)
    ? (r.base_ofv - r.final_ofv) : null;
  return (
    <div>
      <div style={{ fontSize: 12, color: 'var(--text-dim)', marginBottom: 6 }}>
        {r.label} · {r.n_candidates} candidate{r.n_candidates === 1 ? '' : 's'} tested ·
        {' '}forward p&lt;{r.forward_p} / backward p&lt;{r.backward_p} ·
        {' '}OFV {fmt(r.base_ofv ?? undefined, 1)} → {fmt(r.final_ofv ?? undefined, 1)}
        {dOfv != null ? ` (ΔOFV ${fmt(dOfv, 1)})` : ''}
      </div>
      <div style={{ fontSize: 12, marginBottom: 6 }}>
        <strong>Selected:</strong>{' '}
        {selected.length
          ? selected.map(s => `${s.param}~${s.covariate} (${s.kind})`).join(', ')
          : 'none — no covariate met the entry criterion'}
      </div>
      {steps.length > 0 && (
        <table className="nca-table">
          <thead><tr><th>Step</th><th>Effect</th><th>ΔOFV</th><th>χ²crit (df)</th><th>Decision</th></tr></thead>
          <tbody>
            {steps.map((s, i) => (
              <tr key={i}>
                <td>{s.phase}</td>
                <td>{s.effect}</td>
                <td>{fmt(s.delta_ofv, 2)}</td>
                <td>{fmt(s.crit, 2)} ({s.df})</td>
                <td style={{ color: s.decision === 'added' ? 'var(--accent)' : 'var(--text-dim)' }}>
                  {s.decision}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
      {r.final?.covariate_effects && r.final.covariate_effects.length > 0 && (
        <div style={{ fontSize: 11, color: 'var(--text-dim)', marginTop: 6 }}>
          Final effects — {r.final.covariate_effects.map((ce, i) => (
            <span key={i}>{i > 0 ? ' · ' : ''}{ce.param}: {ce.description}
              {typeof ce.rse_pct === 'number' ? ` (${fmt(ce.rse_pct, 0)}% RSE)` : ''}</span>
          ))}
        </div>
      )}
      {r.note ? (
        <div style={{ fontSize: 11, color: 'var(--text-dim)', marginTop: 4 }}>{r.note}</div>
      ) : null}
    </div>
  );
}

function ForecastCard({ r }: { r: PharmState['forecast_results'] }) {
  if (!r) return null;
  if (r.status !== 'ok') {
    return <div className="qc-card conditional"><div className="qc-title">MAP forecast — not run</div>
      <div style={{ fontSize: 12 }}>{r.message}</div></div>;
  }
  const ind = r.individual_params ?? {};
  const typ = r.typical_params ?? {};
  const si = r.ss_individual ?? {};
  const sp = r.ss_population ?? {};
  const rec = r.recommendation;
  const fc = r.forecast;
  const measured = r.measured ?? [];
  const METRICS: [string, string][] = [['cmin', 'Cmin'], ['cmax', 'Cmax'], ['cavg', 'Cavg'], ['auc_tau', 'AUCτ']];

  // inline SVG: individual (solid) + population (dashed) + measured points
  let chart = null;
  if (fc && fc.times.length) {
    const W = 460, H = 150, ml = 40, mr = 10, mt = 10, mb = 22;
    const t = fc.times, yi = fc.individual, yp = fc.population;
    const tmax = Math.max(...t), ymax = Math.max(...yi, ...yp, ...measured.map(m => m.conc), 1e-6);
    const sx = (x: number) => ml + (x / tmax) * (W - ml - mr);
    const sy = (y: number) => H - mb - (y / ymax) * (H - mt - mb);
    const path = (ys: number[]) => t.map((x, i) => `${sx(x).toFixed(1)},${sy(ys[i]).toFixed(1)}`).join(' ');
    chart = (
      <svg viewBox={`0 0 ${W} ${H}`} style={{ width: '100%', maxWidth: W, marginTop: 6 }}>
        <line x1={ml} y1={H - mb} x2={W - mr} y2={H - mb} stroke="var(--border)" />
        <line x1={ml} y1={mt} x2={ml} y2={H - mb} stroke="var(--border)" />
        <polyline points={path(yp)} fill="none" stroke="var(--text-dim)" strokeWidth="1.4" strokeDasharray="4 3" />
        <polyline points={path(yi)} fill="none" stroke="var(--accent)" strokeWidth="1.8" />
        {measured.map((m, i) => (
          <circle key={i} cx={sx(m.time)} cy={sy(m.conc)} r="3.2" fill="var(--green)" stroke="#fff" strokeWidth="0.5" />
        ))}
        <text x={(ml + W - mr) / 2} y={H - 4} textAnchor="middle" fontSize="9" fill="var(--text-dim)">time (h)</text>
      </svg>
    );
  }

  return (
    <div>
      <div style={{ fontSize: 12, color: 'var(--text-dim)', marginBottom: 6 }}>
        {r.label} · MAP from {r.n_obs} measured level{r.n_obs === 1 ? '' : 's'} · {r.dose} q{r.tau}h · wt {r.wt}kg
      </div>
      <table className="nca-table">
        <thead><tr><th>Parameter</th><th>Individual (MAP)</th><th>Population</th></tr></thead>
        <tbody>
          {Object.keys(ind).map(p => (
            <tr key={p}>
              <td>{p}</td>
              <td style={{ color: 'var(--accent)' }}>{fmt(ind[p], 3)}</td>
              <td>{fmt(typ[p], 3)}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <div style={{ fontSize: 11, color: 'var(--text-dim)', marginTop: 6 }}>
        Steady-state exposure (individual vs population):{' '}
        {METRICS.filter(([k]) => si[k] != null).map(([k, lbl], i) => (
          <span key={k}>{i > 0 ? ' · ' : ''}{lbl} {fmt(si[k], 3)} / {fmt(sp[k] ?? undefined, 3)}</span>
        ))}
      </div>
      {chart}
      {chart && (
        <div style={{ fontSize: 11, color: 'var(--text-dim)' }}>
          <span style={{ color: 'var(--accent)' }}>—</span> individual ·{' '}
          <span style={{ color: 'var(--text-dim)' }}>– –</span> population ·{' '}
          <span style={{ color: 'var(--green)' }}>●</span> measured
        </div>
      )}
      {rec && (
        <div style={{ fontSize: 12, marginTop: 8, padding: '8px 10px', background: 'var(--accent-bg)',
          border: '1px solid var(--accent-glow)', borderRadius: 8 }}>
          {rec.recommended_dose != null ? (
            <>Recommended dose to hit <strong>{rec.target_metric} = {rec.target}</strong>:{' '}
              <strong style={{ color: 'var(--accent)' }}>{fmt(rec.recommended_dose, 1)}</strong>
              {rec.predicted ? ` → predicted ${rec.target_metric} ${fmt(rec.predicted[rec.target_metric], 3)}` : ''}</>
          ) : (
            <>Target {rec.target_metric} = {rec.target}: {rec.note}</>
          )}
        </div>
      )}
    </div>
  );
}

// Small reusable residual-vs-x scatter panel, shared by the legacy two-stage
// IWRES plot and the NLME-provenance grid (IWRES/CWRES/npd x PRED/TIME/TAD).
// `tad` arrays carry `null` for observations before any dose — those pairs
// are dropped rather than plotted at a fabricated x=0.
function residualScatterSVG(
  x: (number | null | undefined)[], y: (number | null | undefined)[],
  xlabel: string, ylabel: string, refKey: string,
) {
  const pairs: [number, number][] = [];
  for (let i = 0; i < Math.min(x.length, y.length); i++) {
    const xi = x[i], yi = y[i];
    if (xi != null && yi != null && Number.isFinite(xi) && Number.isFinite(yi)) pairs.push([xi, yi]);
  }
  if (!pairs.length) return null;
  const W = 168, H = 132, m = 26;
  const xmax = Math.max(...pairs.map(p => p[0])) * 1.05 || 1;
  const yabs = Math.max(2, ...pairs.map(p => Math.abs(p[1]))) * 1.1;
  const sx = (v: number) => m + (v / xmax) * (W - m - 6);
  const sy = (v: number) => (H - m) / 2 + 4 - (v / yabs) * ((H - m - 10) / 2);
  return (
    <svg key={refKey} viewBox={`0 0 ${W} ${H}`} width="150px" role="img" aria-label={`${ylabel} vs ${xlabel}`}>
      <line x1={m} y1={sy(0)} x2={W - 6} y2={sy(0)} stroke="var(--text-dim)" strokeDasharray="2 2" />
      <line x1={m} y1={9} x2={m} y2={H - m} stroke="var(--border)" />
      {pairs.map(([xi, yi], i) => (
        <circle key={i} cx={sx(xi)} cy={sy(yi)} r="1.7"
          fill={Math.abs(yi) > 1.96 ? 'var(--yellow)' : 'var(--accent)'} fillOpacity="0.6" />
      ))}
      <text x={(m + W) / 2} y={H - 4} textAnchor="middle" fontSize="8.5" fill="var(--text-dim)">{xlabel}</text>
      <text x={8} y={(9 + H - m) / 2} textAnchor="middle" fontSize="8.5" fill="var(--text-dim)"
        transform={`rotate(-90 8 ${(9 + H - m) / 2})`}>{ylabel}</text>
    </svg>
  );
}

// Distribution histogram with an N(0,1) overlay, shared by every residual row.
function residualHistSVG(y: (number | null | undefined)[], label: string, refKey: string) {
  const vals = y.filter((v): v is number => v != null && Number.isFinite(v));
  if (!vals.length) return null;
  const W = 168, H = 132, m = 26;
  const bins = 11, lo = -3.25, hi = 3.25, bw = (hi - lo) / bins;
  const counts = new Array(bins).fill(0);
  vals.forEach(v => { const b = Math.min(bins - 1, Math.max(0, Math.floor((v - lo) / bw))); counts[b]++; });
  const cmax = Math.max(...counts, 1);
  const bx = (i: number) => m + (i / bins) * (W - m - 6);
  const bwid = (W - m - 6) / bins;
  const by = (c: number) => H - m - (c / cmax) * (H - m - 10);
  const norm = (z: number) => Math.exp(-z * z / 2) / Math.sqrt(2 * Math.PI);
  const peak = norm(0) * vals.length * bw;
  const curve = Array.from({ length: 31 }, (_, k) => {
    const z = lo + (k / 30) * (hi - lo);
    return `${k ? 'L' : 'M'}${bx((z - lo) / bw).toFixed(1)} ${by(norm(z) * vals.length * bw / peak * cmax).toFixed(1)}`;
  }).join(' ');
  return (
    <svg key={refKey} viewBox={`0 0 ${W} ${H}`} width="150px" role="img" aria-label={`${label} distribution`}>
      {counts.map((c, i) => <rect key={i} x={bx(i) + 1} y={by(c)} width={bwid - 2} height={H - m - by(c)}
        fill="var(--accent)" fillOpacity="0.3" />)}
      <path d={curve} fill="none" stroke="var(--green)" strokeWidth="1.3" />
      <line x1={m} y1={H - m} x2={W - 6} y2={H - m} stroke="var(--border)" />
      <text x={(m + W) / 2} y={H - 4} textAnchor="middle" fontSize="8.5" fill="var(--text-dim)">{label} (vs N(0,1))</text>
    </svg>
  );
}

function DiagnosticsCard({ r }: { r: PharmState['diagnostics_results'] }) {
  if (!r || r.status !== 'ok') {
    return <div className="qc-card conditional"><div className="qc-title">Diagnostics — not run</div>
      <div style={{ fontSize: 12 }}>{r?.message}</div></div>;
  }
  const res = r.residuals;
  const cw = r.cwres, np = r.npde;
  // Single-provenance grid (IWRES/CWRES/npd, all from the SAME converged NLME
  // fit) renders only when both blocks are available — otherwise a figure
  // would mix panels from two different estimators. `status` present on
  // either block (needs_nlme / blq_unsupported) means it isn't.
  const gridAvailable = !!(cw && !cw.status && np && !np.status);

  // Legacy two-stage IWRES panel (unweighted log residual): always shown when
  // present, since it needs only a structural fit, not NLME.
  const legacyIwres = res && res.ipred.length
    ? residualScatterSVG(res.ipred, res.iwres, 'IPRED', 'log residual (two-stage)', 'legacy-iwres')
    : null;

  const npdLine = np?.status
    ? `npd unavailable — ${np.message ?? np.status}`
    : `npd mean ${fmt(np?.summary?.mean ?? undefined, 3)} sd ${fmt(np?.summary?.sd ?? undefined, 2)} · `
      + `${fmt(np?.summary?.pct_outside_1_96 ?? undefined, 1)}% outside ±1.96 (ideal ~5%)`;
  const cwresLine = cw?.status
    ? `CWRES unavailable — ${cw.message ?? cw.status}`
    : `CWRES mean ${fmt(cw?.summary?.cwres_mean ?? undefined, 3)} sd ${fmt(cw?.summary?.cwres_sd ?? undefined, 2)} `
      + `(${cw?.summary?.cwres_variant ?? 'focei'})`;

  return (
    <div>
      <div style={{ fontSize: 12, color: 'var(--text-dim)', marginBottom: 6 }}>
        {r.label} · two-stage IWRES mean {fmt(res?.summary.iwres_mean ?? undefined, 3)} sd {fmt(res?.summary.iwres_sd ?? undefined, 2)} ·
        {' '}{cwresLine} · {npdLine}
      </div>
      {!gridAvailable && (
        <>
          <div style={{ fontSize: 11, color: 'var(--text-dim)', marginBottom: 6 }}>
            Run a population fit (run_nlme) on {r.label} to unlock the CWRES/npd grid below (vs PRED/TIME/TAD).
          </div>
          <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap' }}>{legacyIwres}</div>
        </>
      )}
      {gridAvailable && cw && np && (
        <>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, auto)', gap: 8, marginBottom: 8 }}>
            {residualScatterSVG(cw.ipred ?? [], cw.iwres ?? [], 'IPRED', 'IWRES', 'iwres-pred')}
            {residualScatterSVG(cw.time ?? [], cw.iwres ?? [], 'time (h)', 'IWRES', 'iwres-time')}
            {residualScatterSVG(cw.tad ?? [], cw.iwres ?? [], 'TAD (h)', 'IWRES', 'iwres-tad')}

            {residualScatterSVG(cw.cpred ?? [], cw.cwres ?? [], 'CPRED', 'CWRES', 'cwres-pred')}
            {residualScatterSVG(cw.time ?? [], cw.cwres ?? [], 'time (h)', 'CWRES', 'cwres-time')}
            {residualScatterSVG(cw.tad ?? [], cw.cwres ?? [], 'TAD (h)', 'CWRES', 'cwres-tad')}

            {residualScatterSVG(np.pred ?? [], np.npde ?? [], 'sim. median', 'npd', 'npd-pred')}
            {residualScatterSVG(np.time ?? [], np.npde ?? [], 'time (h)', 'npd', 'npd-time')}
            {residualScatterSVG(np.tad ?? [], np.npde ?? [], 'TAD (h)', 'npd', 'npd-tad')}
          </div>
          <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap' }}>
            {residualHistSVG(cw.iwres ?? [], 'IWRES', 'iwres-hist')}
            {residualHistSVG(cw.cwres ?? [], 'CWRES', 'cwres-hist')}
            {residualHistSVG(np.npde ?? [], 'npd', 'npd-hist')}
          </div>
        </>
      )}
    </div>
  );
}

function ForestCard({ r }: { r: PharmState['forest_results'] }) {
  if (!r || r.status !== 'ok') {
    return <div className="qc-card conditional"><div className="qc-title">Covariate forest — not run</div>
      <div style={{ fontSize: 12 }}>{r?.message}</div></div>;
  }
  const rows = r.rows ?? [];
  if (!rows.length) {
    return <div style={{ fontSize: 12, color: 'var(--text-dim)' }}>
      {r.label} ({r.source}) — no covariate effects in the fitted model.
    </div>;
  }
  const W = 560, rowH = 26, top = 26, left = 190, right = 90;
  const H = top + rows.length * rowH + 24;
  // Log-scale x-axis over GMR/CI (x_range already spans 0.9x-1.1x the data,
  // widened further to include a bounds band if present and outside it).
  let [xlo, xhi] = r.x_range ?? [0.5, 2.0];
  if (r.bounds) { xlo = Math.min(xlo, r.bounds[0] * 0.9); xhi = Math.max(xhi, r.bounds[1] * 1.1); }
  const lnLo = Math.log(Math.max(xlo, 1e-6)), lnHi = Math.log(Math.max(xhi, xlo * 1.01));
  const sx = (v: number) => left + ((Math.log(Math.max(v, 1e-6)) - lnLo) / (lnHi - lnLo)) * (W - left - right);
  const ticks = [xlo, xlo * Math.sqrt(xhi / xlo), 1.0, xhi / Math.sqrt(xhi / xlo), xhi]
    .filter((v, i, a) => v > 0 && a.indexOf(v) === i)
    .sort((a, b) => a - b);

  return (
    <div>
      <div style={{ fontSize: 12, color: 'var(--text-dim)', marginBottom: 6 }}>
        {r.label} ({r.source}) · {r.summary?.n_rows} row(s) across {r.summary?.n_effects} effect(s) ·
        {' '}{Math.round((r.ci_level ?? 0.9) * 100)}% CI
      </div>
      <svg viewBox={`0 0 ${W} ${H}`} width="100%" style={{ maxWidth: W }} role="img" aria-label="Covariate forest plot">
        {r.bounds && (
          <rect x={sx(r.bounds[0])} y={top - 10} width={Math.max(0, sx(r.bounds[1]) - sx(r.bounds[0]))}
            height={rows.length * rowH + 14} fill="var(--text-dim)" fillOpacity="0.08" />
        )}
        <line x1={sx(1.0)} y1={top - 10} x2={sx(1.0)} y2={top + rows.length * rowH + 4}
          stroke="var(--text-dim)" strokeDasharray="3 3" />
        {ticks.map((t, i) => (
          <text key={i} x={sx(t)} y={top + rows.length * rowH + 18} textAnchor="middle"
            fontSize="9" fill="var(--text-dim)">{t.toFixed(t < 1 ? 2 : 1)}</text>
        ))}
        {rows.map((row, i) => {
          const y = top + i * rowH + rowH / 2;
          const unavailable = row.gmr == null;
          return (
            <g key={i}>
              <text x={4} y={y + 3} fontSize="10" fill="var(--text)">{row.eval_label}</text>
              {unavailable ? (
                <text x={left} y={y + 3} fontSize="9.5" fill="var(--yellow)">
                  unavailable ({row.ci_source})
                </text>
              ) : (
                <>
                  {row.ci_lo != null && row.ci_hi != null && (
                    <line x1={sx(row.ci_lo)} y1={y} x2={sx(row.ci_hi)} y2={y}
                      stroke={row.outside_reference_band ? 'var(--yellow)' : 'var(--accent)'} strokeWidth="1.6" />
                  )}
                  <circle cx={sx(row.gmr as number)} cy={y} r="3.2"
                    fill={row.outside_reference_band ? 'var(--yellow)' : 'var(--accent)'} />
                  <text x={W - right + 6} y={y + 3} fontSize="9.5" fill="var(--text-dim)">
                    {(row.gmr as number).toFixed(2)}
                    {row.ci_lo != null && row.ci_hi != null ? ` [${row.ci_lo.toFixed(2)}, ${row.ci_hi.toFixed(2)}]` : ''}
                  </text>
                </>
              )}
            </g>
          );
        })}
      </svg>
      {!!r.notes?.length && (
        <ul style={{ fontSize: 11, color: 'var(--text-dim)', margin: '4px 0 0', paddingLeft: 16 }}>
          {r.notes.map((n, i) => <li key={i}>{n}</li>)}
        </ul>
      )}
    </div>
  );
}

/** Reusable pcVPC / dose-normalized VPC SVG — used for both the main pcVPC and
 * the per-stratum grid. Observed 5/50/95 (solid green) over the simulated
 * 5/50/95 band (dashed) plus the simulated-median 90% CI ribbon. */
function pcvpcSvg(bins: PcVpcBin[] | undefined, opts?: {
  width?: number; height?: number; xLabel?: string; ariaLabel?: string;
  yMax?: number;
}) {
  if (!bins) return null;
  const pts = bins.filter(b => b.t != null);
  if (!pts.length) return null;
  const PW = opts?.width ?? 580, PH = opts?.height ?? 230, pm = 44, pr = 12, pt = 12, pb = 28;
  const ts = pts.map(b => b.t as number);
  const tmin = Math.min(...ts), tmax = Math.max(...ts);
  const vals = pts.flatMap(b => [b.obs_p95, b.sim_p95, b.sim_med_hi]).filter(v => v != null) as number[];
  const cmax = opts?.yMax ?? ((Math.max(...vals) || 1) * 1.05);
  const sx = (v: number) => pm + ((v - tmin) / (tmax - tmin || 1)) * (PW - pm - pr);
  const sy = (v: number) => PH - pb - (v / cmax) * (PH - pt - pb);
  const linePts = (key: 'obs_p05' | 'obs_p50' | 'obs_p95' | 'sim_p05' | 'sim_p50' | 'sim_p95') =>
    pts.filter(b => b[key] != null)
      .map((b, i) => `${i ? 'L' : 'M'}${sx(b.t as number).toFixed(1)} ${sy(b[key] as number).toFixed(1)}`).join(' ');
  const ci = pts.filter(b => b.sim_med_lo != null && b.sim_med_hi != null);
  const up = ci.map(b => `${sx(b.t as number).toFixed(1)},${sy(b.sim_med_hi as number).toFixed(1)}`).join(' ');
  const dn = ci.map(b => `${sx(b.t as number).toFixed(1)},${sy(b.sim_med_lo as number).toFixed(1)}`).reverse().join(' ');
  return (
    <Zoomable title={opts?.ariaLabel ?? "pcVPC"}>
    <svg viewBox={`0 0 ${PW} ${PH}`} style={{ width: '100%', maxWidth: PW, marginTop: 8 }}
      role="img" aria-label={opts?.ariaLabel ?? 'Prediction-corrected VPC'}>
      {ci.length > 1 && <polygon points={`${up} ${dn}`} fill="var(--accent)" fillOpacity="0.18" />}
      {(['sim_p05', 'sim_p95'] as const).map(k =>
        <path key={k} d={linePts(k)} fill="none" stroke="var(--text-dim)" strokeWidth="1" strokeDasharray="4 3" />)}
      <path d={linePts('sim_p50')} fill="none" stroke="var(--accent)" strokeWidth="1.4" strokeDasharray="4 3" />
      {(['obs_p05', 'obs_p95'] as const).map(k =>
        <path key={k} d={linePts(k)} fill="none" stroke="var(--green)" strokeWidth="1.1" />)}
      <path d={linePts('obs_p50')} fill="none" stroke="var(--green)" strokeWidth="1.9" />
      {pts.filter(b => b.obs_p50 != null).map((b, i) =>
        <circle key={i} cx={sx(b.t as number)} cy={sy(b.obs_p50 as number)} r="2.4" fill="var(--green)" />)}
      <line x1={pm} y1={PH - pb} x2={PW - pr} y2={PH - pb} stroke="var(--border)" />
      <line x1={pm} y1={pt} x2={pm} y2={PH - pb} stroke="var(--border)" />
      <text x={(pm + PW) / 2} y={PH - 6} textAnchor="middle" fontSize="10" fill="var(--text-dim)">
        {opts?.xLabel ?? 'time (h)'}</text>
      <text x={12} y={(pt + PH - pb) / 2} textAnchor="middle" fontSize="10" fill="var(--text-dim)"
        transform={`rotate(-90 12 ${(pt + PH - pb) / 2})`}>prediction-corrected conc.</text>
    </svg>
    </Zoomable>
  );
}

/** Small-multiples grid of per-stratum pcVPC panels, mirroring the flexplot
 * facet layout. A shared y-axis makes the strata directly comparable. */
function StratifiedVpcPanels({ s }: { s: NonNullable<PharmState['vpc_results']>['stratified'] }) {
  if (!s) return null;
  if (s.status !== 'ok' || !s.strata?.length) {
    return <div style={{ fontSize: 12, color: 'var(--yellow)', marginTop: 8 }}>
      Stratified VPC unavailable: {s.message ?? s.status}
      {s.available && <> · available: {s.available.join(', ')}</>}
    </div>;
  }
  const xLabel = s.x_by === 'tad' ? 'time after dose (h)' : 'time (h)';
  const label = (s.stratify_by || 'dose (normalized)');
  // Shared y-domain across panels so the strata are directly comparable.
  const yMax = Math.max(1, ...s.strata.flatMap(st => st.bins.flatMap(
    b => [b.obs_p95, b.sim_p95, b.sim_med_hi]).filter(v => v != null) as number[])) * 1.05;
  const tile = s.strata.length > 1 ? 380 : 560;
  return (
    <div style={{ marginTop: 12 }}>
      <div style={{ fontSize: 12, color: 'var(--text-dim)', marginBottom: 4 }}>
        Prediction-corrected VPC stratified by <b>{label}</b>
        {s.correction === 'dose' && ' · dose-normalized'} · {s.strata.length} strata · shared y-axis
      </div>
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 12 }}>
        {s.strata.map(st => (
          <div key={st.label} style={{ width: tile, maxWidth: '100%' }}>
            <div style={{ fontSize: 11, color: 'var(--text-dim)', fontWeight: 600 }}>
              {label} = {st.label} <span>(n = {st.n})</span>
            </div>
            {pcvpcSvg(st.bins, { width: tile, height: 200, xLabel, yMax,
              ariaLabel: `pcVPC for ${label} = ${st.label}` })}
          </div>
        ))}
      </div>
      {!!s.skipped?.length && (
        <div style={{ fontSize: 11, color: 'var(--text-dim)', marginTop: 4 }}>
          Skipped: {s.skipped.map(k => `${k.label} (${k.reason})`).join(', ')}
        </div>
      )}
    </div>
  );
}

type ExpMetricT = NonNullable<NonNullable<PharmState['vpc_results']>['exposure_pc']>['groups'];

/** One histogram of the simulated group-mean exposure, with the observed mean
 * (solid) and the simulated 2.5-97.5% interval (dashed) overlaid. */
function expHistSvg(g: NonNullable<ExpMetricT>[number], metric: 'auc' | 'cmax', gb: string) {
  const m = g[metric];
  const edges = m.hist.edges, counts = m.hist.counts;
  if (edges.length < 2) return null;
  const W = 250, H = 150, ml = 8, mr = 8, mt = 6, mb = 24;
  const lo = Math.min(edges[0], m.observed), hi = Math.max(edges[edges.length - 1], m.observed);
  const sx = (v: number) => ml + ((v - lo) / (hi - lo || 1)) * (W - ml - mr);
  const cmax = Math.max(1, ...counts);
  const sy = (c: number) => H - mb - (c / cmax) * (H - mt - mb);
  const vline = (v: number | null, color: string, dash: boolean) =>
    v == null ? null :
      <line x1={sx(v)} y1={mt} x2={sx(v)} y2={H - mb} stroke={color}
        strokeWidth={dash ? 1 : 1.7} strokeDasharray={dash ? '4 3' : undefined} />;
  return (
    <div key={g.label} style={{ width: W, maxWidth: '100%' }}>
      <div style={{ fontSize: 11, color: 'var(--text-dim)' }}>
        {gb} = {g.label} <span>(n = {g.n})</span>{' '}
        <span style={{ color: m.within ? 'var(--green)' : 'var(--red, #c0392b)' }}>
          {m.within ? '✓ within' : '✗ outside'}</span>
      </div>
      <Zoomable title="VPC panel">
      <svg viewBox={`0 0 ${W} ${H}`} style={{ width: '100%', maxWidth: W }} role="img"
        aria-label={`Exposure predictive check ${metric} for ${gb} ${g.label}`}>
        {counts.map((c, i) => (
          <rect key={i} x={sx(edges[i])} y={sy(c)} width={Math.max(0.5, sx(edges[i + 1]) - sx(edges[i]) - 0.5)}
            height={H - mb - sy(c)} fill="var(--accent)" fillOpacity="0.5" />
        ))}
        {vline(m.sim_lo, 'var(--text-dim)', true)}
        {vline(m.sim_hi, 'var(--text-dim)', true)}
        {vline(m.observed, 'var(--green)', false)}
        <line x1={ml} y1={H - mb} x2={W - mr} y2={H - mb} stroke="var(--border)" />
        <text x={W / 2} y={H - 4} textAnchor="middle" fontSize="9" fill="var(--text-dim)">
          {metric === 'auc' ? 'mean AUC (conc·h)' : 'mean Cmax (conc)'}</text>
      </svg>
      </Zoomable>
    </div>
  );
}

/** Exposure predictive check: for AUC and Cmax, one simulated-mean histogram per
 * group with the observed mean and the simulated interval overlaid. */
function ExposurePcPanel({ e }: { e: NonNullable<PharmState['vpc_results']>['exposure_pc'] }) {
  if (!e) return null;
  if (e.status !== 'ok' || !e.groups?.length) {
    return <div style={{ fontSize: 12, color: 'var(--yellow)', marginTop: 8 }}>
      Exposure predictive check unavailable: {e.message ?? e.status}</div>;
  }
  const gb = e.group_by || 'group';
  const ci = e.ci && e.ci.length === 2 ? Math.round(e.ci[1] - e.ci[0]) : 95;
  return (
    <div style={{ marginTop: 12 }}>
      <div style={{ fontSize: 12, color: 'var(--text-dim)', marginBottom: 4 }}>
        Exposure predictive check — observed group mean (—) vs simulated-mean distribution
        ({ci}% interval dashed), by <b>{gb}</b>{e.multiple_dose && ' · last-interval exposure'}
      </div>
      {(['auc', 'cmax'] as const).map(metric => (
        <div key={metric} style={{ marginTop: 6 }}>
          <div style={{ fontSize: 11, color: 'var(--text-dim)', fontWeight: 600 }}>
            {metric === 'auc' ? 'Mean AUC' : 'Mean Cmax'}
          </div>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 10 }}>
            {e.groups!.map(g => expHistSvg(g, metric, gb))}
          </div>
        </div>
      ))}
      {!!e.skipped?.length && (
        <div style={{ fontSize: 11, color: 'var(--text-dim)', marginTop: 4 }}>
          Skipped: {e.skipped.map(k => `${k.label} (n = ${k.n}, ${k.reason})`).join(', ')}
        </div>
      )}
    </div>
  );
}

/** BLQ-incidence VPC: observed fraction below LLOQ per bin vs the simulated
 * median and 5-95% band — the categorical companion to the concentration VPC. */
function BlqVpcPanel({ b }: { b: NonNullable<PharmState['vpc_results']>['blq_vpc'] }) {
  if (!b) return null;
  if (b.status !== 'ok' || !b.bins?.length) {
    return <div style={{ fontSize: 12, color: 'var(--yellow)', marginTop: 8 }}>
      BLQ-incidence VPC unavailable: {b.message ?? b.status}</div>;
  }
  const pts = b.bins.filter(p => p.x != null);
  if (!pts.length) return null;
  const W = 580, H = 220, pm = 44, pr = 12, pt = 12, pb = 28;
  const xs = pts.map(p => p.x as number);
  const xmin = Math.min(...xs), xmax = Math.max(...xs);
  const sx = (v: number) => pm + ((v - xmin) / (xmax - xmin || 1)) * (W - pm - pr);
  const sy = (v: number) => H - pb - Math.max(0, Math.min(1, v)) * (H - pt - pb);
  const ci = pts.filter(p => p.sim_lo != null && p.sim_hi != null);
  const up = ci.map(p => `${sx(p.x as number).toFixed(1)},${sy(p.sim_hi as number).toFixed(1)}`).join(' ');
  const dn = ci.map(p => `${sx(p.x as number).toFixed(1)},${sy(p.sim_lo as number).toFixed(1)}`).reverse().join(' ');
  const linePts = (key: 'sim_med' | 'obs_frac') =>
    pts.filter(p => p[key] != null)
      .map((p, i) => `${i ? 'L' : 'M'}${sx(p.x as number).toFixed(1)} ${sy(p[key] as number).toFixed(1)}`).join(' ');
  const xLabel = b.x_by === 'tad' ? 'time after dose (h)' : 'time (h)';
  return (
    <div style={{ marginTop: 12 }}>
      <div style={{ fontSize: 12, color: 'var(--text-dim)', marginBottom: 4 }}>
        BLQ-incidence VPC — fraction below LLOQ ({b.lloq}) over {xLabel}; {b.n_blq} censored obs
      </div>
      <svg viewBox={`0 0 ${W} ${H}`} style={{ width: '100%', maxWidth: W }}
        role="img" aria-label="BLQ-incidence VPC">
        {ci.length > 1 && <polygon points={`${up} ${dn}`} fill="var(--accent)" fillOpacity="0.18" />}
        <path d={linePts('sim_med')} fill="none" stroke="var(--accent)" strokeWidth="1.4" strokeDasharray="4 3" />
        <path d={linePts('obs_frac')} fill="none" stroke="var(--green)" strokeWidth="1.9" />
        {pts.filter(p => p.obs_frac != null).map((p, i) =>
          <circle key={i} cx={sx(p.x as number)} cy={sy(p.obs_frac as number)} r="2.4" fill="var(--green)" />)}
        {[0, 0.5, 1].map((f, i) => (
          <g key={i}>
            <line x1={pm} y1={sy(f)} x2={W - pr} y2={sy(f)} stroke="var(--border)" strokeOpacity="0.5" />
            <text x={pm - 6} y={sy(f) + 3} textAnchor="end" fontSize="9" fill="var(--text-dim)">
              {(f * 100).toFixed(0)}%</text>
          </g>
        ))}
        <line x1={pm} y1={pt} x2={pm} y2={H - pb} stroke="var(--border)" />
        <text x={(pm + W) / 2} y={H - 6} textAnchor="middle" fontSize="10" fill="var(--text-dim)">{xLabel}</text>
        <text x={12} y={(pt + H - pb) / 2} textAnchor="middle" fontSize="10" fill="var(--text-dim)"
          transform={`rotate(-90 12 ${(pt + H - pb) / 2})`}>fraction &lt; LLOQ</text>
      </svg>
      <div style={{ fontSize: 11, color: 'var(--text-dim)', display: 'flex', gap: 14 }}>
        <span><span style={{ color: 'var(--green)' }}>—</span> observed fraction BLQ</span>
        <span><span style={{ color: 'var(--accent)' }}>– –</span> simulated median</span>
        <span style={{ color: 'var(--accent)' }}>▦ simulated 5–95%</span>
      </div>
    </div>
  );
}

function VpcCard({ r, onRerun, busy, covariates }: {
  r: PharmState['vpc_results'];
  onRerun?: (o: { stratify_by?: string | null; dose_normalize?: boolean; x_by?: string;
    exposure_check?: boolean; blq_check?: boolean }) => void;
  busy?: boolean;
  covariates?: string[];
}) {
  // Controls state is seeded from the run that produced this card, so the knobs
  // reflect what is actually plotted (each rerun mounts a fresh VpcCard).
  const [stratifyBy, setStratifyBy] = useState(() => r?.stratified?.stratify_by ?? '');
  const [doseNorm, setDoseNorm] = useState(() => r?.stratified?.correction === 'dose');
  const [xTad, setXTad] = useState(() => r?.stratified?.x_by === 'tad');
  const [expCheck, setExpCheck] = useState(() => !!r?.exposure_pc);
  const [blqCheck, setBlqCheck] = useState(() => !!r?.blq_vpc);
  if (!r || r.status !== 'ok') {
    return <div className="qc-card conditional"><div className="qc-title">VPC / GOF — not run</div>
      <div style={{ fontSize: 12 }}>{r?.message}</div></div>;
  }
  const ovp = r.obs_vs_pred;
  const vpc = r.vpc;
  // obs-vs-pred scatter (observed x, ipred y) with identity line
  const W = 280, H = 240, m = 38;
  let scatter = null;
  if (ovp && ovp.observed.length) {
    const obs = ovp.observed, ip = ovp.ipred;
    const hi = Math.max(...obs, ...ip) * 1.05 || 1;
    const sx = (v: number) => m + (v / hi) * (W - m - 8);
    const sy = (v: number) => H - m - (v / hi) * (H - m - 8);
    scatter = (
      <Zoomable title="Observed vs predicted" style={{ width: '48%', maxWidth: W }}>
      <svg viewBox={`0 0 ${W} ${H}`} style={{ width: '100%' }} role="img" aria-label="Observed vs predicted">
        <line x1={sx(0)} y1={sy(0)} x2={sx(hi)} y2={sy(hi)} stroke="var(--text-dim)" strokeDasharray="3 3" />
        <line x1={m} y1={H - m} x2={W - 8} y2={H - m} stroke="var(--border)" />
        <line x1={m} y1={8} x2={m} y2={H - m} stroke="var(--border)" />
        {obs.map((o, i) => <circle key={i} cx={sx(o)} cy={sy(ip[i])} r="2.2" fill="var(--accent)" fillOpacity="0.6" />)}
        <text x={(m + W) / 2} y={H - 6} textAnchor="middle" fontSize="10" fill="var(--text-dim)">observed</text>
        <text x={11} y={(8 + H - m) / 2} textAnchor="middle" fontSize="10" fill="var(--text-dim)"
          transform={`rotate(-90 11 ${(8 + H - m) / 2})`}>predicted (IPRED)</text>
      </svg>
      </Zoomable>
    );
  }
  // VPC band
  let band = null;
  if (vpc && vpc.times.length) {
    const t = vpc.times, p05 = vpc.p05, p50 = vpc.p50, p95 = vpc.p95;
    const tmin = Math.min(...t), tmax = Math.max(...t);
    const allC = [...p95, ...(r.obs_c ?? [])];
    const cmax = Math.max(...allC) * 1.05 || 1;
    const sx = (v: number) => m + ((v - tmin) / (tmax - tmin || 1)) * (W - m - 8);
    const sy = (v: number) => H - m - (v / cmax) * (H - m - 8);
    const up = t.map((x, i) => `${sx(x).toFixed(1)},${sy(p95[i]).toFixed(1)}`).join(' ');
    const dn = t.map((x, i) => `${sx(x).toFixed(1)},${sy(p05[i]).toFixed(1)}`).reverse().join(' ');
    const med = t.map((x, i) => `${i ? 'L' : 'M'}${sx(x).toFixed(1)} ${sy(p50[i]).toFixed(1)}`).join(' ');
    band = (
      <Zoomable title="Visual predictive check" style={{ width: '48%', maxWidth: W }}>
      <svg viewBox={`0 0 ${W} ${H}`} style={{ width: '100%' }} role="img" aria-label="Visual predictive check">
        <polygon points={`${up} ${dn}`} fill="var(--accent)" fillOpacity="0.16" />
        <path d={med} fill="none" stroke="var(--accent)" strokeWidth="1.4" />
        {(r.obs_t ?? []).map((x, i) => <circle key={i} cx={sx(x)} cy={sy((r.obs_c ?? [])[i])} r="1.8" fill="var(--green)" fillOpacity="0.65" />)}
        <line x1={m} y1={H - m} x2={W - 8} y2={H - m} stroke="var(--border)" />
        <line x1={m} y1={8} x2={m} y2={H - m} stroke="var(--border)" />
        <text x={(m + W) / 2} y={H - 6} textAnchor="middle" fontSize="10" fill="var(--text-dim)">time (h)</text>
        <text x={11} y={(8 + H - m) / 2} textAnchor="middle" fontSize="10" fill="var(--text-dim)"
          transform={`rotate(-90 11 ${(8 + H - m) / 2})`}>conc.</text>
      </svg>
      </Zoomable>
    );
  }
  // prediction-corrected VPC — rendered by the shared pcvpcSvg helper, which
  // is reused for the per-stratum small-multiples below (single drawing idiom).
  const pc = r.pcvpc;
  const pcXLabel = pc?.x_by === 'tad' ? 'time after dose (h)' : 'time (h)';
  const pcChart = pc && pc.status === 'ok' ? pcvpcSvg(pc.bins, { xLabel: pcXLabel }) : null;

  return (
    <div>
      <div style={{ fontSize: 12, color: 'var(--text-dim)', marginBottom: 6 }}>
        {r.label} · GOF log-scale R²(IPRED) = {fmt(r.gof?.r2_log_ipred ?? undefined, 3)} ·
        RMSE = {fmt(r.gof?.rmse_log_ipred ?? undefined, 3)} · n = {r.gof?.n}
        {r.vpc_dose != null && ` · VPC @ dose ${r.vpc_dose}`}
      </div>
      <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap' }}>{scatter}{band}</div>
      <div style={{ fontSize: 11, color: 'var(--text-dim)', display: 'flex', gap: 14, marginTop: 2 }}>
        <span><span style={{ color: 'var(--accent)' }}>—</span> predicted (median + 5–95% band)</span>
        <span><span style={{ color: 'var(--green)' }}>•</span> observed</span>
      </div>
      {pcChart && (
        <>
          <div style={{ fontSize: 12, color: 'var(--text-dim)', margin: '10px 0 0' }}>
            Prediction-corrected VPC — {pc?.n_bins} time bins, {pc?.n_sim} simulations
          </div>
          {pcChart}
          <div style={{ fontSize: 11, color: 'var(--text-dim)', display: 'flex', gap: 14 }}>
            <span><span style={{ color: 'var(--green)' }}>—</span> observed 5/50/95</span>
            <span><span style={{ color: 'var(--accent)' }}>– –</span> simulated 5/50/95</span>
            <span style={{ color: 'var(--accent)' }}>▦ simulated-median 90% CI</span>
          </div>
        </>
      )}
      {onRerun && (
        <div style={{ display: 'flex', gap: 12, alignItems: 'center', flexWrap: 'wrap',
          margin: '12px 0 0', paddingTop: 10, borderTop: '1px solid var(--border)', fontSize: 12 }}>
          <span style={{ color: 'var(--text-dim)' }}>Pooling across dose groups misleads —
            stratify or dose-normalize:</span>
          <label style={{ color: 'var(--text-dim)' }}>
            by{' '}
            <select className="model-select" style={{ maxWidth: 150 }} value={stratifyBy} disabled={busy}
              onChange={e => setStratifyBy(e.target.value)}>
              <option value="">none (pooled)</option>
              {(covariates ?? []).map(c => <option key={c} value={c}>{c}</option>)}
            </select>
          </label>
          <label style={{ color: 'var(--text-dim)', display: 'inline-flex', gap: 4, alignItems: 'center' }}>
            <input type="checkbox" checked={doseNorm} disabled={busy}
              onChange={e => setDoseNorm(e.target.checked)} /> dose-normalize
          </label>
          <label style={{ color: 'var(--text-dim)', display: 'inline-flex', gap: 4, alignItems: 'center' }}>
            <input type="checkbox" checked={xTad} disabled={busy}
              onChange={e => setXTad(e.target.checked)} /> time-after-dose
          </label>
          <label style={{ color: 'var(--text-dim)', display: 'inline-flex', gap: 4, alignItems: 'center' }}
            title="Observed group-mean AUC/Cmax vs the simulated-mean distribution">
            <input type="checkbox" checked={expCheck} disabled={busy}
              onChange={e => setExpCheck(e.target.checked)} /> exposure PC
          </label>
          <label style={{ color: 'var(--text-dim)', display: 'inline-flex', gap: 4, alignItems: 'center' }}
            title="Fraction of observations below the LLOQ over time vs the simulated band (needs censored data)">
            <input type="checkbox" checked={blqCheck} disabled={busy}
              onChange={e => setBlqCheck(e.target.checked)} /> BLQ VPC
          </label>
          <button className="chip" disabled={busy}
            onClick={() => onRerun({ stratify_by: stratifyBy || null, dose_normalize: doseNorm,
              x_by: xTad ? 'tad' : 'time', exposure_check: expCheck, blq_check: blqCheck })}>
            {busy ? 'Running…' : 'Recompute VPC'}
          </button>
        </div>
      )}
      {r.stratified && <StratifiedVpcPanels s={r.stratified} />}
      {r.exposure_pc && <ExposurePcPanel e={r.exposure_pc} />}
      {r.blq_vpc && <BlqVpcPanel b={r.blq_vpc} />}
    </div>
  );
}

export {
  PkPopTable,
  PkModelCard,
  PriorCheckCard,
  NlmeCard,
  EngineComparisonCard,
  ScmCard,
  ForecastCard,
  DiagnosticsCard,
  ForestCard,
  VpcCard,
};
