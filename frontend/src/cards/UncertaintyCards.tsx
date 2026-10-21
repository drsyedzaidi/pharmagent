import type { PharmState, ProfileParam, SimestReplicate } from '../types';
import { UncertaintyNotes } from './shared';
import { ci, fmt } from './helpers';

function SimestReplicatePlot({ replicates, param }: { replicates: SimestReplicate[]; param: string }) {
  const pts = replicates
    .map(r => ({ theta: r.theta[param], ci: r.ci?.[param] ?? null }))
    .filter(p => p.theta != null);
  if (!pts.length) return null;
  const W = 260, rowH = 22, top = 8, left = 8, right = 8;
  const H = top + pts.length * rowH + 18;
  const allVals = pts.flatMap(p => (p.ci ? [p.ci[0], p.ci[1]] : [p.theta]));
  const lo = Math.min(...allVals) * 0.95, hi = Math.max(...allVals) * 1.05;
  const sx = (v: number) => left + ((v - lo) / Math.max(hi - lo, 1e-9)) * (W - left - right);
  return (
    <svg viewBox={`0 0 ${W} ${H}`} width="100%" style={{ maxWidth: W }} role="img"
      aria-label={`${param} across replicates`}>
      {pts.map((p, i) => {
        const y = top + i * rowH + rowH / 2;
        return (
          <g key={i}>
            {p.ci && <line x1={sx(p.ci[0])} y1={y} x2={sx(p.ci[1])} y2={y} stroke="var(--accent)" strokeWidth="1.6" />}
            <circle cx={sx(p.theta)} cy={y} r="3" fill="var(--accent)" />
          </g>
        );
      })}
      <text x={left} y={H - 4} fontSize="9" fill="var(--text-dim)">{lo.toFixed(2)}</text>
      <text x={W - right} y={H - 4} fontSize="9" fill="var(--text-dim)" textAnchor="end">{hi.toFixed(2)}</text>
    </svg>
  );
}

function SimestCard({ r }: { r: PharmState['simest_results'] }) {
  if (!r || !['ok', 'partial', 'not_evaluable'].includes(r.status)) {
    return <div className="qc-card conditional"><div className="qc-title">Trial-design check — not run</div>
      <div style={{ fontSize: 12 }}>{r?.message}</div></div>;
  }
  const params = r.params ?? [];
  return (
    <div>
      <div style={{ fontSize: 12, color: 'var(--text-dim)', marginBottom: 6 }}>
        {r.n_rep_completed}/{r.n_rep_planned} replicate(s) completed · {r.n_point_evaluable} point-evaluable ·
        {' '}{r.n_ci_evaluable} CI-evaluable ({r.ci_validity}) ·
        {' '}strict pass rate {r.criterion?.pct_within_60_140_strict}%
        {r.criterion?.target_pct != null && (
          <> vs target {r.criterion.target_pct}% — {r.criterion.criterion_met ? 'MET' : 'NOT MET'}</>
        )}
      </div>
      <div style={{ overflowX: 'auto' }}>
        <table style={{ fontSize: 11, borderCollapse: 'collapse', width: '100%' }}>
          <thead>
            <tr style={{ color: 'var(--text-dim)', textAlign: 'left' }}>
              <th>param</th><th>truth</th><th>GM est.</th><th>bias%</th><th>RMSE%</th>
              <th>CV%</th><th>pass (strict)</th><th>coverage 95% CI</th>
            </tr>
          </thead>
          <tbody>
            {params.map(p => {
              const s = r.per_param?.[p];
              if (!s) return null;
              return (
                <tr key={p} style={{ borderTop: '1px solid var(--border)' }}>
                  <td>{p}</td>
                  <td>{fmt(s.truth ?? undefined, 3)}</td>
                  <td>{fmt(s.gm_point_estimate ?? undefined, 3)}</td>
                  <td>{fmt(s.rel_bias_pct ?? undefined, 1)}</td>
                  <td>{fmt(s.rmse_pct ?? undefined, 1)}</td>
                  <td>{fmt(s.cv_across_replicates_pct ?? undefined, 1)}</td>
                  <td>{fmt(s.pct_within_60_140_strict ?? undefined, 0)}%</td>
                  <td>[{fmt(s.coverage_wilson_ci_pct?.[0] ?? undefined, 0)}, {fmt(s.coverage_wilson_ci_pct?.[1] ?? undefined, 0)}]</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      {!!r.replicates?.length && (
        <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap', marginTop: 8 }}>
          {params.map(p => (
            <div key={p}>
              <div style={{ fontSize: 11, color: 'var(--text-dim)' }}>{p} per replicate</div>
              <SimestReplicatePlot replicates={r.replicates!} param={p} />
            </div>
          ))}
        </div>
      )}
      {!!r.design_limitations?.length && (
        <ul style={{ fontSize: 11, color: 'var(--text-dim)', margin: '8px 0 0', paddingLeft: 16 }}>
          {r.design_limitations.map((n, i) => <li key={i}>{n}</li>)}
          {r.citation && <li>{r.citation}</li>}
        </ul>
      )}
    </div>
  );
}

function BootstrapCard({ r }: { r: PharmState['bootstrap_results'] }) {
  if (!r) return null;
  if (r.status !== 'ok') {
    return <div className="qc-card conditional"><div className="qc-title">Bootstrap — not run ({r.status})</div>
      <div style={{ fontSize: 12 }}>{r.message}</div></div>;
  }
  const rate = r.success_rate != null ? Math.round(100 * r.success_rate) : null;
  const lowRate = rate != null && rate < 80;
  const ratio = new Map((r.comparison ?? []).map(c => [c.parameter, c.width_ratio_boot_over_asymptotic]));
  const last = r.stability?.length ? r.stability[r.stability.length - 1] : null;
  const prev = r.stability && r.stability.length > 1 ? r.stability[r.stability.length - 2] : null;
  return (
    <div>
      <div style={{ fontSize: 12, color: 'var(--text-dim)', marginBottom: 6 }}>
        {r.method} · {r.n_ok}/{r.n_completed} replicates converged
        {rate != null && <span style={{ color: lowRate ? 'var(--red)' : 'inherit' }}> ({rate}%{lowRate ? ' ⚠' : ''})</span>}
        {' '}of {r.n_boot_requested} requested · {r.n_subjects} subjects
        {r.stratified ? ` · stratified (${r.n_strata})` : ' · unstratified'}
        {' '}· {Math.round((r.ci_level ?? 0.95) * 100)}% percentile CI · {fmt(r.seconds, 0)}s
        {r.stopped_early && <span style={{ color: 'var(--yellow)' }}> · stopped at budget</span>}
      </div>
      <table className="nca-table">
        <thead><tr><th>Parameter</th><th>Estimate</th><th>Boot median</th><th>Boot CI</th><th>Boot SE</th>
          <th>Bias%</th><th>Asymptotic CI</th><th title="bootstrap width / asymptotic width — >1 means the asymptotic interval is optimistic (too narrow)">Width ratio</th></tr></thead>
        <tbody>
          {(r.parameters ?? []).map(p => {
            const wr = ratio.get(p.parameter);
            return (
              <tr key={p.parameter}>
                <td>{p.parameter}</td>
                <td>{fmt(p.estimate ?? undefined, 3)}</td>
                <td>{fmt(p.boot_median ?? undefined, 3)}</td>
                <td>{ci(p.boot_lo, p.boot_hi)}</td>
                <td>{fmt(p.boot_se ?? undefined, 3)}</td>
                <td>{fmt(p.boot_bias_pct ?? undefined, 1)}</td>
                <td>{ci(p.asymptotic_lo, p.asymptotic_hi)}</td>
                <td style={{ color: wr != null && wr > 1.25 ? 'var(--yellow)' : 'inherit' }}>
                  {wr != null ? fmt(wr, 2) : '–'}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
      {last && (
        <div style={{ fontSize: 11, color: 'var(--text-dim)', marginTop: 6 }}>
          CI stability — at {last.n_replicates} replicates:{' '}
          {last.parameters.map(p => `${p.parameter} ${ci(p.lo, p.hi)}`).join(' · ')}
          {prev && <> · at {prev.n_replicates}: {prev.parameters.map(p => `${p.parameter} ${ci(p.lo, p.hi)}`).join(' · ')}</>}
        </div>
      )}
      <UncertaintyNotes notes={r.notes} />
    </div>
  );
}

function SirCard({ r }: { r: PharmState['sir_results'] }) {
  if (!r) return null;
  if (r.status !== 'ok') {
    return <div className="qc-card conditional"><div className="qc-title">SIR — not run ({r.status})</div>
      <div style={{ fontSize: 12 }}>{r.message}</div></div>;
  }
  const d = r.diagnostics;
  const lowEss = d?.ess_fraction_of_m != null && d.ess_fraction_of_m < 0.2;
  return (
    <div>
      <div style={{ fontSize: 12, color: 'var(--text-dim)', marginBottom: 6 }}>
        {r.method} · M {r.n_samples} ({r.n_usable} usable) → m {r.n_resample} (M/m {fmt(r.m_over_m_ratio, 1)})
        {' '}· inflation {fmt(r.inflation, 2)} · {Math.round((r.ci_level ?? 0.95) * 100)}% CI · {fmt(r.seconds, 0)}s
        {r.stopped_early && <span style={{ color: 'var(--yellow)' }}> · stopped at budget</span>}
      </div>
      <table className="nca-table">
        <thead><tr><th>Parameter</th><th>Estimate</th><th>SIR median</th><th>SIR CI</th><th>Asymptotic CI</th>
          <th title="(upper − estimate) / (estimate − lower); 1 = symmetric">Asymmetry</th></tr></thead>
        <tbody>
          {(r.parameters ?? []).map(p => (
            <tr key={p.parameter}>
              <td>{p.parameter}</td>
              <td>{fmt(p.estimate ?? undefined, 3)}</td>
              <td>{fmt(p.sir_median ?? undefined, 3)}</td>
              <td>{ci(p.sir_lo, p.sir_hi)}</td>
              <td>{ci(p.asymptotic_lo, p.asymptotic_hi)}</td>
              <td>{p.asymmetry_ratio != null ? fmt(p.asymmetry_ratio, 2) : '–'}</td>
            </tr>
          ))}
        </tbody>
      </table>
      {d && (
        <div style={{ fontSize: 11, color: 'var(--text-dim)', marginTop: 6 }}>
          ESS <span style={{ color: lowEss ? 'var(--red)' : 'inherit' }}>{fmt(d.effective_sample_size ?? undefined, 0)}
          {' '}({fmt((d.ess_fraction_of_m ?? 0) * 100, 0)}% of m){lowEss ? ' ⚠' : ''}</span>
          {' '}· resampled dOFV mean {fmt(d.dofv_mean_resampled ?? undefined, 1)} vs df {d.df_reference ?? '–'}
          {' '}· proposal dOFV mean {fmt(d.dofv_mean_proposal ?? undefined, 1)} · max weight {fmt(d.max_weight ?? undefined, 3)}
        </div>
      )}
      <UncertaintyNotes notes={r.notes} />
    </div>
  );
}

function ProfileSparkline({ p, cutoff }: { p: ProfileParam; cutoff: number }) {
  const pts = p.profile ?? [];
  if (pts.length < 2) return null;
  const W = 150, H = 48, pad = 4;
  const xs = pts.map(q => q.value), ys = pts.map(q => q.dofv);
  const x0 = Math.min(...xs), x1 = Math.max(...xs);
  const yMax = Math.max(cutoff * 1.2, ...ys.filter(Number.isFinite));
  const yMin = Math.min(0, ...ys.filter(Number.isFinite));
  const sx = (v: number) => pad + ((v - x0) / (x1 - x0 || 1)) * (W - 2 * pad);
  const sy = (v: number) => H - pad - ((v - yMin) / (yMax - yMin || 1)) * (H - 2 * pad);
  const path = pts.map((q, i) => `${i ? 'L' : 'M'}${sx(q.value).toFixed(1)},${sy(q.dofv).toFixed(1)}`).join(' ');
  return (
    <svg width={W} height={H} style={{ display: 'block' }}>
      <line x1={pad} x2={W - pad} y1={sy(cutoff)} y2={sy(cutoff)} stroke="var(--yellow)" strokeDasharray="3 2" />
      <line x1={pad} x2={W - pad} y1={sy(0)} y2={sy(0)} stroke="var(--border)" />
      <path d={path} fill="none" stroke="var(--accent)" strokeWidth={1.5} />
      {p.estimate != null && <line x1={sx(p.estimate)} x2={sx(p.estimate)} y1={pad} y2={H - pad} stroke="var(--text-dim)" strokeDasharray="2 2" />}
    </svg>
  );
}

function ProfileCard({ r }: { r: PharmState['profile_results'] }) {
  if (!r) return null;
  if (r.status !== 'ok') {
    return <div className="qc-card conditional"><div className="qc-title">Likelihood profiling — not run ({r.status})</div>
      <div style={{ fontSize: 12 }}>{r.message}</div></div>;
  }
  const d = r.diagnostics;
  const notOpt = !!d?.fit_not_at_optimum && (!Array.isArray(d.fit_not_at_optimum) || d.fit_not_at_optimum.length > 0);
  const cutoff = r.dofv_cutoff ?? 3.84;
  return (
    <div>
      <div style={{ fontSize: 12, color: 'var(--text-dim)', marginBottom: 6 }}>
        {r.method} · {Math.round((r.ci_level ?? 0.95) * 100)}% CI at dOFV {fmt(cutoff, 2)} · {r.n_parameters} parameter(s)
        {' '}· {r.n_evaluations} evaluations · {fmt(r.seconds, 0)}s
      </div>
      {notOpt && (
        <div className="qc-card fail" style={{ marginBottom: 6 }}>
          <div className="qc-title">Fit was not at an optimum</div>
          <div style={{ fontSize: 12 }}>
            The profile found a lower OFV than the reported fit
            {Array.isArray(d!.fit_not_at_optimum) && d!.fit_not_at_optimum.length ? ` for ${d!.fit_not_at_optimum.join(', ')}` : ''}.
            Re-fit (e.g. method=auto) before trusting any interval.
          </div>
        </div>
      )}
      {!!d?.non_monotone_parameters?.length && (
        <div style={{ fontSize: 11, color: 'var(--yellow)', marginBottom: 4 }}>
          Non-monotone profile: {d.non_monotone_parameters.join(', ')} — multiple optima or a rough objective.
        </div>
      )}
      {!!d?.unbounded_parameters?.length && (
        <div style={{ fontSize: 11, color: 'var(--yellow)', marginBottom: 4 }}>
          Limit not reached (unbounded): {d.unbounded_parameters.join(', ')}.
        </div>
      )}
      <table className="nca-table">
        <thead><tr><th>Parameter</th><th>Estimate</th><th>Profile CI</th><th>Lower</th><th>Upper</th>
          <th title="(upper − estimate) / (estimate − lower); 1 = symmetric">Asymmetry</th><th>Evals</th><th>dOFV profile</th></tr></thead>
        <tbody>
          {(r.parameters ?? []).map(p => (
            <tr key={p.parameter}>
              <td>{p.parameter}</td>
              <td>{fmt(p.estimate ?? undefined, 3)}</td>
              <td>{ci(p.profile_lo, p.profile_hi)}</td>
              <td style={{ fontSize: 11, color: p.lower_reason ? 'var(--yellow)' : 'var(--text-dim)' }}>{p.lower_reason ?? 'ok'}</td>
              <td style={{ fontSize: 11, color: p.upper_reason ? 'var(--yellow)' : 'var(--text-dim)' }}>{p.upper_reason ?? 'ok'}</td>
              <td>{p.asymmetry_ratio != null ? fmt(p.asymmetry_ratio, 2) : '–'}</td>
              <td>{p.n_evaluations}</td>
              <td><ProfileSparkline p={p} cutoff={cutoff} /></td>
            </tr>
          ))}
        </tbody>
      </table>
      <UncertaintyNotes notes={r.notes} />
    </div>
  );
}

export {
  SimestReplicatePlot,
  SimestCard,
  BootstrapCard,
  SirCard,
  ProfileSparkline,
  ProfileCard,
};
