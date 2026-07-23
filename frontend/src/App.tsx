import { useEffect, useRef, useState, useCallback } from 'react';
import type { DragEvent, ChangeEvent } from 'react';
import {
  FlaskConical, Upload, Send, CheckCircle, XCircle,
  FileText, ShieldCheck, AlertTriangle, ChevronRight,
  Download, Loader2, Activity,
} from 'lucide-react';
import { api, setToken, getToken } from './api';
import { FlexplotPanel } from './flexplot';
import type {
  Session, PharmState, AgentMessage, AuditEntry,
  WorkflowStatus, ContentBlock, PkModelDef, ReviewResults, ReviewFinding, Severity, SkillDef,
  SpaghettiData, NcaPlotData, LzSubject, SimestReplicate, WorkflowResponse,
  PcVpcBin,
} from './types';

const agentColor: Record<string, string> = {
  supervisor: '#1F66A6', data_manager: '#3B86C9',
  nca: '#1D7A5A', qc: '#9A5B12', report: '#4A6FA5',
};
const agentLabel: Record<string, string> = {
  supervisor: 'Supervisor', data_manager: 'Data Manager',
  nca: 'NCA Agent', qc: 'QC Agent', report: 'Report Agent',
};

const STEPS = [
  { key: 'load_dataset',       label: 'Load dataset' },
  { key: 'profile_pk_dataset', label: 'Profile PK data' },
  { key: 'validate_cdisc',     label: 'Validate format' },
  { key: 'spaghetti_plot',     label: 'Spaghetti plot' },
  { key: 'compute_nca',        label: 'Compute NCA' },
  { key: 'adversarial_review', label: 'Adversarial review' },
  { key: 'qc_review',          label: 'QC review',  gate: true },
  { key: 'generate_report',    label: 'Generate report' },
] as const;

const MODELING_STEPS = [
  { key: 'load_dataset',          label: 'Load dataset' },
  { key: 'profile_pk_dataset',    label: 'Profile PK data' },
  { key: 'fit_pk_model',          label: 'Fit structural models' },
  { key: 'run_engine_comparison', label: 'Cross-engine comparison' },
  { key: 'adversarial_review',    label: 'Adversarial review', gate: true },
] as const;

const POPPK_FULL_STEPS = [
  { key: 'load_dataset',         label: 'Load dataset' },
  { key: 'profile_pk_dataset',   label: 'Profile PK data' },
  { key: 'validate_cdisc',       label: 'Validate format' },
  { key: 'spaghetti_plot',       label: 'Spaghetti plot' },
  { key: 'fit_pk_model',         label: 'Compare structural models', gate: true },
  { key: 'run_nlme',             label: 'Population (NLME) fit' },
  { key: 'run_scm',              label: 'Covariate model (SCM)' },
  { key: 'run_diagnostics',      label: 'Residual diagnostics' },
  { key: 'run_covariate_forest', label: 'Covariate forest' },
  { key: 'run_vpc',              label: 'VPC / goodness-of-fit' },
  { key: 'adversarial_review',   label: 'Adversarial review', gate: true },
  { key: 'generate_report',      label: 'Generate report' },
] as const;

type WorkflowName = 'nca_full' | 'poppk_modeling' | 'poppk_full';

/** Sidebar presentation per workflow — keeps the step tracker in one place. */
const WORKFLOW_UI: Record<WorkflowName, { title: string; steps: readonly { key: string; label: string; gate?: boolean }[] }> = {
  nca_full:       { title: 'NCA Workflow',        steps: STEPS },
  poppk_modeling: { title: 'Modeling Workflow',   steps: MODELING_STEPS },
  poppk_full:     { title: 'Population PK Workflow', steps: POPPK_FULL_STEPS },
};

/** Steps that submit a real population fit — minutes of compute, so resuming
 *  into one is polled as a background job rather than awaited inline. */
const HEAVY_STEPS = new Set<string>([
  'run_nlme', 'run_scm', 'run_engine_comparison', 'run_simest',
]);

const WF_LABEL: Record<WorkflowName, string> = {
  nca_full: 'NCA',
  poppk_modeling: 'population modeling',
  poppk_full: 'full population PK',
};

function fmt(v: number | undefined, d = 2) {
  if (v == null || isNaN(v)) return '–';
  return v.toFixed(d);
}

// `snap` freezes the PharmState slice a result card reads, so re-running an
// analysis later does not retroactively rewrite earlier cards in the transcript.
type DisplayMsg = { role: string; content: string; agent?: string; tool?: string; id: string; snap?: PharmState | null };

function MessageBubble({ msg, agent }: { msg: DisplayMsg; agent?: string }) {
  const isUser = msg.role === 'user';
  return (
    <div className={`msg ${isUser ? 'user' : 'agent'}`}>
      <div className="msg-avatar" style={!isUser ? { color: agentColor[agent ?? ''] ?? 'var(--accent)' } : {}}>
        {isUser ? 'You' : (agent ?? 'AI').slice(0, 2).toUpperCase()}
      </div>
      <div className="msg-bubble">
        {!isUser && agent && (
          <div className="msg-agent-tag" style={{ color: agentColor[agent] ?? 'var(--accent)' }}>
            {agentLabel[agent] ?? agent}
          </div>
        )}
        <div style={{ whiteSpace: 'pre-wrap' }}>{msg.content}</div>
        {msg.tool && (
          <div className="tool-chip">
            <ChevronRight size={10} /> {msg.tool}
          </div>
        )}
      </div>
    </div>
  );
}

function QcCard({ state }: { state: PharmState }) {
  const v = state.qc_verdict ?? '';
  const cls = v === 'PASS' ? 'pass' : v.includes('CONDITIONAL') ? 'conditional' : 'fail';
  return (
    <div className={`qc-card ${cls}`}>
      <div className="qc-title">
        {cls === 'pass' && <CheckCircle size={14} style={{ display: 'inline', marginRight: 6 }} />}
        {cls === 'fail' && <XCircle size={14} style={{ display: 'inline', marginRight: 6 }} />}
        {cls === 'conditional' && <AlertTriangle size={14} style={{ display: 'inline', marginRight: 6 }} />}
        QC Verdict: {v}
      </div>
      {state.qc_checklist && state.qc_checklist.length > 0 && (
        <ul className="qc-issues" style={{ listStyle: 'none', padding: 0 }}>
          {state.qc_checklist.map((c, i) => (
            <li key={i}>
              {c.status === 'PASS' ? '✓' : c.status === 'FAIL' ? '✗' : '!'} {c.check}
              <span style={{ opacity: 0.7 }}> — {c.detail}</span>
            </li>
          ))}
        </ul>
      )}
      {state.qc_issues && state.qc_issues.length > 0 && (
        <ul className="qc-issues">
          {state.qc_issues.map((iss, i) => (
            <li key={i}>[{iss.severity}] {iss.issue}</li>
          ))}
        </ul>
      )}
    </div>
  );
}

const SEVERITY_COLOR: Record<Severity, string> = {
  CRITICAL: '#B23A2E', HIGH: '#9A5B12', MEDIUM: '#1F66A6', LOW: '#8298AC',
};

function ReviewCard({ r }: { r: ReviewResults }) {
  const c = r.counts;
  return (
    <div className="review-card">
      <div className="review-goal"
        style={{ color: r.goal_met ? '#1D7A5A' : '#9A5B12', fontWeight: 600 }}>
        {r.goal_met
          ? <><CheckCircle size={14} style={{ display: 'inline', marginRight: 6 }} />Goal met</>
          : <><AlertTriangle size={14} style={{ display: 'inline', marginRight: 6 }} />Findings block the goal</>}
        <span style={{ opacity: 0.6, fontWeight: 400 }}> — {r.goal}</span>
      </div>
      <div className="review-counts" style={{ display: 'flex', gap: 10, margin: '6px 0 10px' }}>
        {(['CRITICAL', 'HIGH', 'MEDIUM', 'LOW'] as Severity[]).map(s => (
          <span key={s} style={{ fontSize: 12, color: SEVERITY_COLOR[s], opacity: c[s] ? 1 : 0.4 }}>
            {c[s]} {s.toLowerCase()}
          </span>
        ))}
      </div>
      {r.findings.length === 0 ? (
        <div style={{ opacity: 0.7, fontSize: 13 }}>
          No findings — the reviewer could not refute any reported value.
        </div>
      ) : (
        <ul style={{ listStyle: 'none', padding: 0, margin: 0, display: 'flex', flexDirection: 'column', gap: 8 }}>
          {r.findings.map((f: ReviewFinding) => (
            <li key={f.id} style={{ borderLeft: `3px solid ${SEVERITY_COLOR[f.severity]}`, paddingLeft: 10 }}>
              <div style={{ fontSize: 12 }}>
                <span style={{ color: SEVERITY_COLOR[f.severity], fontWeight: 700 }}>{f.severity}</span>
                <span style={{ opacity: 0.85 }}> · {f.target}</span>
              </div>
              <div style={{ fontSize: 13, marginTop: 2 }}><strong>Claim:</strong> {f.claim}</div>
              <div style={{ fontSize: 13, opacity: 0.9 }}><strong>Evidence:</strong> {f.evidence}</div>
              <div style={{ fontSize: 12, opacity: 0.7, marginTop: 2 }}>→ {f.suggested_action}</div>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

function SkillsPanel({ skills, loading, datasetPath, onRun, onDelete, onMarkdown }: {
  skills: SkillDef[]; loading: boolean; datasetPath: string | null;
  onRun: (name: string) => void; onDelete: (name: string) => void;
  onMarkdown: (name: string) => void;
}) {
  return (
    <div className="quick-actions" style={{ flexDirection: 'column', alignItems: 'stretch' }}>
      {skills.length === 0 ? (
        <div style={{ opacity: 0.7, fontSize: 13 }}>
          No skills captured yet. Run an analysis, then “Capture as skill”.
        </div>
      ) : (
        <ul style={{ listStyle: 'none', padding: 0, margin: 0, display: 'flex', flexDirection: 'column', gap: 6 }}>
          {skills.map(s => (
            <li key={s.name} style={{ display: 'flex', alignItems: 'center', gap: 8, flexWrap: 'wrap' }}>
              <span style={{ fontWeight: 600 }}>{s.name}</span>
              <span style={{ fontSize: 12, opacity: 0.6 }}>v{s.version}</span>
              <span style={{ fontSize: 12, opacity: 0.7 }}>{s.steps.map(st => st.tool).join(' → ')}</span>
              <span style={{ flex: 1 }} />
              <button className="chip" disabled={loading || !datasetPath} onClick={() => onRun(s.name)}
                title={datasetPath ? 'Replay on the current dataset' : 'Load a dataset first'}>Replay</button>
              <button className="chip" disabled={loading} onClick={() => onMarkdown(s.name)}>SKILL.md</button>
              <button className="chip" disabled={loading} onClick={() => onDelete(s.name)}>Delete</button>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

function NcaSubjectTable({ state }: { state: PharmState }) {
  const rows = state.nca_parameters;
  if (!rows || rows.length === 0) return null;
  const ss = state.nca_summary?.steady_state === true;
  const tau = rows.find(r => r.tau != null)?.tau;

  const summ = state.nca_summary;
  const meta = summ ? `${summ.route ?? 'extravascular'}${summ.blq ? ` · ${summ.blq.n_below_loq} BLQ` : ''}` : '';
  if (ss) {
    return (
      <div>
        <div style={{ fontSize: 12, color: 'var(--text-dim)', marginBottom: 4 }}>
          Steady-state NCA — {rows.length} subjects · AUC over τ{tau ? ` = ${tau} h` : ''}{meta && ` · ${meta}`}
        </div>
        <table className="nca-table">
          <thead>
            <tr>
              <th>ID</th><th>Dose</th><th>Cmax,ss</th><th>Cmin</th>
              <th>AUC<sub>τ</sub></th><th>Cavg</th>
              <th>CL/F</th><th>t½</th><th>Fluct%</th><th>R<sub>ac</sub></th>
            </tr>
          </thead>
          <tbody>
            {rows.map(r => (
              <tr key={String(r.subject)}>
                <td>{r.subject}</td>
                <td>{fmt(r.dose, 0)}</td>
                <td>{fmt(r.Cmax)}</td>
                <td>{fmt(r.Cmin ?? undefined)}</td>
                <td>{fmt(r.AUC_tau ?? r.AUC_last, 1)}</td>
                <td>{fmt(r.Cavg ?? undefined, 1)}</td>
                <td>{fmt(r.CL_F, 2)}</td>
                <td>{fmt(r.t_half, 1)}</td>
                <td>{fmt(r.fluctuation_pct ?? undefined, 0)}</td>
                <td>{fmt(r.accumulation_ratio ?? undefined, 2)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    );
  }

  return (
    <div>
      <div style={{ fontSize: 12, color: 'var(--text-dim)', marginBottom: 4 }}>
        Per-subject NCA — {rows.length} subjects{meta && ` · ${meta}`}
      </div>
      <table className="nca-table">
        <thead>
          <tr>
            <th>ID</th><th>Dose</th><th>Cmax</th><th>Tmax</th>
            <th>AUC<sub>last</sub></th><th>AUC<sub>inf</sub></th>
            <th>t½</th><th>CL/F</th><th>Vz/F</th><th>%extrap</th>
          </tr>
        </thead>
        <tbody>
          {rows.map(r => {
            const hi = (r.pct_AUC_extrap ?? 0) > 20;
            return (
              <tr key={String(r.subject)}>
                <td>{r.subject}</td>
                <td>{fmt(r.dose, 0)}</td>
                <td>{fmt(r.Cmax)}</td>
                <td>{fmt(r.Tmax)}</td>
                <td>{fmt(r.AUC_last, 1)}</td>
                <td>{fmt(r.AUC_inf ?? undefined, 1)}</td>
                <td>{fmt(r.t_half, 1)}</td>
                <td>{fmt(r.CL_F, 2)}</td>
                <td>{fmt(r.Vz_F, 1)}</td>
                <td style={hi ? { color: 'var(--yellow)' } : {}}>{fmt(r.pct_AUC_extrap ?? undefined, 1)}%</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

function DoseSummaryTable({ state }: { state: PharmState }) {
  const s = state.nca_summary;
  if (!s || s.by_dose.length === 0) return null;
  // Weight-based dosing yields many near-unique doses (coincidental ties aside).
  // Only show a dose-group summary when there are genuinely few dose levels with
  // replicates — i.e. CV/geomean across the group is meaningful.
  const fewLevels = s.by_dose.length <= Math.max(3, Math.floor(s.n_subjects / 3));
  const hasReplicates = s.by_dose.some(d => d.n >= 3);
  if (!fewLevels || !hasReplicates) return null;
  return (
    <div style={{ marginTop: 12 }}>
      <div style={{ fontSize: 12, color: 'var(--text-dim)', marginBottom: 4 }}>
        Dose-group summary (geometric mean · geoCV%)
      </div>
      <table className="nca-table">
        <thead>
          <tr>
            <th>Dose</th><th>N</th><th>Cmax</th><th>geoCV%</th>
            <th>AUC<sub>inf</sub></th><th>geoCV%</th><th>t½ (median)</th>
          </tr>
        </thead>
        <tbody>
          {s.by_dose.map(row => (
            <tr key={row.dose}>
              <td>{fmt(row.dose, 0)}</td>
              <td>{row.n}</td>
              <td>{fmt(row.Cmax_geomean)}</td>
              <td>{row.Cmax_geocv_pct == null ? '–' : fmt(row.Cmax_geocv_pct, 1) + '%'}</td>
              <td>{fmt(row.AUC_inf_geomean, 1)}</td>
              <td>{row.AUC_inf_geocv_pct == null ? '–' : fmt(row.AUC_inf_geocv_pct, 1) + '%'}</td>
              <td>{fmt(row.t_half_median, 1)} h</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function NcaTable({ state }: { state: PharmState }) {
  return (
    <div>
      <NcaSubjectTable state={state} />
      <DoseSummaryTable state={state} />
    </div>
  );
}

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
      <div style={{ fontSize: 11, opacity: 0.85, marginBottom: 6 }}>
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
      <div style={{ fontSize: 11, opacity: 0.85, marginBottom: 6 }}>
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

function fmtIIV(cv: number | null): { text: string; unstable: boolean } {
  if (cv == null) return { text: '–', unstable: false };
  if (cv > 300) return { text: '≫300% ⚠', unstable: true };
  return { text: fmt(cv, 1) + '%', unstable: false };
}

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
        <div style={{ fontSize: 10, color: 'var(--text-dim)' }}>
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
        <ul style={{ fontSize: 10.5, color: 'var(--text-dim)', margin: '4px 0 0', paddingLeft: 16 }}>
          {r.notes.map((n, i) => <li key={i}>{n}</li>)}
        </ul>
      )}
    </div>
  );
}

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
              <div style={{ fontSize: 10.5, color: 'var(--text-dim)' }}>{p} per replicate</div>
              <SimestReplicatePlot replicates={r.replicates!} param={p} />
            </div>
          ))}
        </div>
      )}
      {!!r.design_limitations?.length && (
        <ul style={{ fontSize: 10.5, color: 'var(--text-dim)', margin: '8px 0 0', paddingLeft: 16 }}>
          {r.design_limitations.map((n, i) => <li key={i}>{n}</li>)}
          {r.citation && <li>{r.citation}</li>}
        </ul>
      )}
    </div>
  );
}

const SPAG_PALETTE = ['#1F66A6','#1D7A5A','#9A5B12','#4A6FA5','#B23A2E','#3B86C9','#16604A','#C77F2A','#5E7388','#2A8F8F'];

function SpaghettiChart({ data }: { data: SpaghettiData }) {
  const [logY, setLogY] = useState(data.log_scale);
  const [individual, setIndividual] = useState(false);

  const W = 500, H = 200, ml = 44, mr = 10, mt = 12, mb = 28;
  const allY = data.series.flatMap(s => s.y).filter(v => v > 0);
  const allX = data.series.flatMap(s => s.x).filter(isFinite);
  if (!allY.length || !allX.length) return <div style={{ fontSize: 12, color: 'var(--text-dim)' }}>No data to plot.</div>;

  const xmax = Math.max(...allX) * 1.05;
  const ymin = Math.min(...allY), ymax = Math.max(...allY) * 1.1;
  const cw = W - ml - mr, ch = H - mt - mb;
  const sx = (x: number) => ml + (x / xmax) * cw;
  const lmin10 = Math.log10(ymin * 0.8), lmax10 = Math.log10(ymax);
  const syLog = (y: number) => y > 0 ? H - mb - (Math.log10(y) - lmin10) / (lmax10 - lmin10) * ch : H - mb;
  const syLin = (y: number) => H - mb - (y / ymax) * ch;
  const sy = logY ? syLog : syLin;

  const toggleBtn = (active: boolean, label: string, onClick: () => void) => (
    <button onClick={onClick} style={{
      fontSize: 10, padding: '2px 8px', borderRadius: 10, cursor: 'pointer', marginLeft: 4,
      border: '1px solid var(--border)',
      background: active ? 'var(--accent)' : 'transparent',
      color: active ? '#fff' : 'var(--text-dim)',
    }}>{label}</button>
  );

  const controls = (
    <div style={{ display: 'flex', alignItems: 'center', marginBottom: 4, flexWrap: 'wrap', gap: 2 }}>
      <span style={{ fontSize: 10, color: 'var(--text-dim)', marginRight: 4 }}>Y axis:</span>
      {toggleBtn(logY, 'Log', () => setLogY(true))}
      {toggleBtn(!logY, 'Linear', () => setLogY(false))}
      <span style={{ fontSize: 10, color: 'var(--text-dim)', marginLeft: 10, marginRight: 4 }}>View:</span>
      {toggleBtn(!individual, 'Overlay', () => setIndividual(false))}
      {toggleBtn(individual, 'Individual', () => setIndividual(true))}
    </div>
  );

  if (individual) {
    const sw = 165, sh = 120, sml = 32, smr = 5, smt = 8, smb = 22;
    return (
      <div>
        {controls}
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 5 }}>
          {data.series.map((s, i) => {
            const color = SPAG_PALETTE[i % SPAG_PALETTE.length];
            const pts = s.x.map((x, j) => ({ x, y: s.y[j] })).filter(p => p.y > 0);
            if (!pts.length) return null;
            const ax = Math.max(...pts.map(p => p.x)) * 1.05;
            const ays = pts.map(p => p.y);
            const aln = Math.log10(Math.min(...ays) * 0.8), alx = Math.log10(Math.max(...ays) * 1.1);
            const ach = sh - smt - smb;
            const scx = (x: number) => sml + (x / ax) * (sw - sml - smr);
            const scy = (y: number) => logY
              ? (y > 0 ? sh - smb - (Math.log10(y) - aln) / (alx - aln) * ach : sh - smb)
              : sh - smb - (y / Math.max(...ays) / 1.1) * ach;
            const poly = pts.map(p => `${scx(p.x).toFixed(1)},${scy(p.y).toFixed(1)}`).join(' ');
            return (
              <svg key={s.id} viewBox={`0 0 ${sw} ${sh}`} width={sw}
                style={{ background: 'rgba(31,102,166,0.03)', borderRadius: 4, border: '1px solid var(--border)' }}>
                <line x1={sml} y1={sh - smb} x2={sw - smr} y2={sh - smb} stroke="var(--border)" />
                <line x1={sml} y1={smt} x2={sml} y2={sh - smb} stroke="var(--border)" />
                {pts.length > 1 && <polyline points={poly} fill="none" stroke={color} strokeWidth="1.3" strokeOpacity="0.8" />}
                {pts.map((p, j) => <circle key={j} cx={scx(p.x)} cy={scy(p.y)} r="2" fill={color} />)}
                <text x={(sml + sw - smr) / 2} y={sh - 4} textAnchor="middle" fontSize="9" fill="var(--text-dim)">{s.id}</text>
              </svg>
            );
          })}
        </div>
        {data.blq_excluded > 0 && (
          <div style={{ fontSize: 10, color: 'var(--text-dim)', marginTop: 4 }}>{data.blq_excluded} BLQ (≤0) excluded</div>
        )}
      </div>
    );
  }

  // Y axis ticks
  const yTicks = logY
    ? (() => {
        const lo = Math.floor(lmin10), hi = Math.ceil(lmax10);
        return Array.from({ length: hi - lo + 1 }, (_, k) => Math.pow(10, lo + k))
          .filter(v => { const yy = syLog(v); return yy >= mt && yy <= H - mb; });
      })()
    : [0.25, 0.5, 0.75, 1.0].map(f => f * ymax);

  const xTicks = [0.25, 0.5, 0.75, 1.0].map(f => Math.round(f * xmax));

  return (
    <div>
      {controls}
      <svg viewBox={`0 0 ${W} ${H}`} style={{ width: '100%', maxWidth: W }}>
        <line x1={ml} y1={H - mb} x2={W - mr} y2={H - mb} stroke="var(--border)" />
        <line x1={ml} y1={mt} x2={ml} y2={H - mb} stroke="var(--border)" />
        {yTicks.map((v, k) => {
          const yy = sy(v);
          const lbl = v >= 1000 ? `${(v / 1000).toFixed(0)}k` : v >= 1 ? String(Math.round(v)) : v.toPrecision(1);
          return (
            <g key={k}>
              <line x1={ml - 3} y1={yy} x2={ml} y2={yy} stroke="var(--text-dim)" />
              <text x={ml - 5} y={yy + 3.5} textAnchor="end" fontSize="8" fill="var(--text-dim)">{lbl}</text>
            </g>
          );
        })}
        {xTicks.map((v, k) => (
          <g key={k}>
            <line x1={sx(v)} y1={H - mb} x2={sx(v)} y2={H - mb + 3} stroke="var(--text-dim)" />
            <text x={sx(v)} y={H - mb + 11} textAnchor="middle" fontSize="8" fill="var(--text-dim)">{v}</text>
          </g>
        ))}
        <text x={(ml + W - mr) / 2} y={H - 3} textAnchor="middle" fontSize="9" fill="var(--text-dim)">{data.x_label}</text>
        <text x={10} y={(mt + H - mb) / 2} textAnchor="middle" fontSize="9" fill="var(--text-dim)"
          transform={`rotate(-90 10 ${(mt + H - mb) / 2})`}>{data.y_label}</text>
        {data.series.map((s, i) => {
          const color = SPAG_PALETTE[i % SPAG_PALETTE.length];
          const pts = s.x.map((x, j) => ({ x, y: s.y[j] })).filter(p => p.y > 0);
          if (!pts.length) return null;
          const poly = pts.map(p => `${sx(p.x).toFixed(1)},${sy(p.y).toFixed(1)}`).join(' ');
          return (
            <g key={s.id}>
              {pts.length > 1 && <polyline points={poly} fill="none" stroke={color} strokeWidth="1.2" strokeOpacity="0.7" />}
              {pts.map((p, j) => <circle key={j} cx={sx(p.x)} cy={sy(p.y)} r="2" fill={color} fillOpacity="0.85" />)}
            </g>
          );
        })}
      </svg>
      <div style={{ fontSize: 10, color: 'var(--text-dim)', marginTop: 2 }}>
        {data.n_subjects} subjects{data.blq_excluded > 0 ? ` · ${data.blq_excluded} BLQ excluded` : ''}
      </div>
    </div>
  );
}

interface LzManualFit {
  lambda_z: number; lambda_z_intercept: number;
  t_half: number; r2_adj: number; n_pts: number;
  lz_x: number[]; lz_y: number[];
  fit_x: number[]; fit_y: number[];
}

function NcaLzPlot({ data, sessionId }: { data: NcaPlotData; sessionId: string }) {
  const subjects = data.subjects;

  const [selections, setSelections] = useState<Record<string, Set<string>>>(() => {
    const s: Record<string, Set<string>> = {};
    for (const sub of subjects) s[sub.id] = new Set(sub.lz_x.map(v => v.toFixed(4)));
    return s;
  });
  const [localFits, setLocalFits] = useState<Record<string, LzManualFit>>({});
  const [loadingId, setLoadingId] = useState<string | null>(null);
  const [errors, setErrors] = useState<Record<string, string>>({});

  const toggle = (sid: string, tk: string) => {
    setSelections(prev => {
      const next = new Set(prev[sid]);
      if (next.has(tk)) next.delete(tk); else next.add(tk);
      return { ...prev, [sid]: next };
    });
    setErrors(prev => ({ ...prev, [sid]: '' }));
  };

  const doRefit = async (s: LzSubject) => {
    const sel = selections[s.id] ?? new Set<string>();
    const pts = s.x.map((x, i) => ({ x, y: s.y[i] }))
      .filter(p => p.y > 0 && sel.has(p.x.toFixed(4)));
    if (pts.length < 3) { setErrors(prev => ({ ...prev, [s.id]: 'Select ≥ 3 points' })); return; }
    setLoadingId(s.id);
    setErrors(prev => ({ ...prev, [s.id]: '' }));
    try {
      const res = await api.refitLz(sessionId, {
        subject: s.id,
        selected_times: pts.map(p => p.x),
        selected_concs: pts.map(p => p.y),
      });
      setLocalFits(prev => ({ ...prev, [s.id]: res }));
    } catch (e) {
      setErrors(prev => ({ ...prev, [s.id]: (e as Error).message }));
    } finally {
      setLoadingId(null);
    }
  };

  const reset = (s: LzSubject) => {
    setSelections(prev => ({ ...prev, [s.id]: new Set(s.lz_x.map(v => v.toFixed(4))) }));
    setLocalFits(prev => { const n = { ...prev }; delete n[s.id]; return n; });
    setErrors(prev => ({ ...prev, [s.id]: '' }));
  };

  if (!subjects.length) return null;
  const sw = 175, sh = 118, sml = 34, smr = 6, smt = 10, smb = 20;

  return (
    <div>
      <div style={{ fontSize: 10, color: 'var(--text-dim)', marginBottom: 6 }}>
        Click points to include/exclude · Refit to apply manual selection
      </div>
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8 }}>
        {subjects.map((s: LzSubject) => {
          const sel = selections[s.id] ?? new Set<string>();
          const fit = localFits[s.id];
          const isLoading = loadingId === s.id;

          const allObs = s.x.map((x, i) => ({ x, y: s.y[i] })).filter(p => p.y > 0);
          const activeFitX = fit ? fit.fit_x : s.fit_x;
          const activeFitY = fit ? fit.fit_y : s.fit_y;
          const activeTHalf = fit ? fit.t_half : s.t_half;
          const activeR2 = fit ? fit.r2_adj : s.r2_adj;
          const activeN = fit ? fit.n_pts : s.n_pts;

          const allY = [...allObs.map(p => p.y), ...activeFitY].filter(v => v > 0);
          const allX = [...allObs.map(p => p.x), ...activeFitX].filter(isFinite);
          if (!allY.length || !allX.length) return null;

          const xmax = Math.max(...allX) * 1.05;
          const lmin = Math.log10(Math.min(...allY) * 0.75);
          const lmax = Math.log10(Math.max(...allY) * 1.3);
          const ach = sh - smt - smb;
          const scx = (x: number) => sml + (x / xmax) * (sw - sml - smr);
          const scy = (y: number) => y > 0 ? sh - smb - (Math.log10(y) - lmin) / (lmax - lmin) * ach : sh - smb;

          const ylo = Math.floor(lmin), yhi = Math.ceil(lmax);
          const yTicks = Array.from({ length: yhi - ylo + 1 }, (_, k) => Math.pow(10, ylo + k))
            .filter(v => { const yy = scy(v); return yy >= smt && yy <= sh - smb; });

          const fitPoly = activeFitX.map((x, i) =>
            `${scx(x).toFixed(1)},${scy(activeFitY[i]).toFixed(1)}`).join(' ');

          const nSel = sel.size;
          const canRefit = nSel >= 3;

          return (
            <div key={s.id} style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 3 }}>
              <svg viewBox={`0 0 ${sw} ${sh}`} width={sw}
                style={{ background: 'rgba(31,102,166,0.03)', borderRadius: 4, border: '1px solid var(--border)', display: 'block' }}>
                <line x1={sml} y1={sh - smb} x2={sw - smr} y2={sh - smb} stroke="var(--border)" />
                <line x1={sml} y1={smt} x2={sml} y2={sh - smb} stroke="var(--border)" />
                {yTicks.map((v, k) => {
                  const yy = scy(v);
                  return (
                    <g key={k}>
                      <line x1={sml - 3} y1={yy} x2={sml} y2={yy} stroke="var(--border)" />
                      <text x={sml - 4} y={yy + 3} textAnchor="end" fontSize="7" fill="var(--text-dim)">
                        {v >= 1000 ? `${(v / 1000).toFixed(0)}k` : v >= 1 ? String(Math.round(v)) : v.toPrecision(1)}
                      </text>
                    </g>
                  );
                })}
                {[0.5, 1.0].map((f, k) => {
                  const v = Math.round(f * xmax);
                  return v > 0 ? (
                    <g key={k}>
                      <line x1={scx(v)} y1={sh - smb} x2={scx(v)} y2={sh - smb + 3} stroke="var(--border)" />
                      <text x={scx(v)} y={sh - smb + 9} textAnchor="middle" fontSize="7" fill="var(--text-dim)">{v}</text>
                    </g>
                  ) : null;
                })}
                {activeFitX.length > 1 && (
                  <polyline points={fitPoly} fill="none" stroke="#9A5B12" strokeWidth="1.4" strokeDasharray="4 3" />
                )}
                {allObs.map((p, j) => {
                  const tk = p.x.toFixed(4);
                  const isSelected = sel.has(tk);
                  const cx = scx(p.x), cy = scy(p.y);
                  return (
                    <g key={j} style={{ cursor: 'pointer' }} onClick={() => toggle(s.id, tk)}>
                      <circle cx={cx} cy={cy} r="7" fill="transparent" />
                      <circle cx={cx} cy={cy} r={isSelected ? 3.5 : 2.5}
                        fill={isSelected ? 'var(--accent)' : 'var(--text-dim)'}
                        fillOpacity={isSelected ? 1 : 0.4}
                        stroke={isSelected ? '#fff' : 'none'} strokeWidth="0.5" />
                    </g>
                  );
                })}
                <text x={(sml + sw - smr) / 2} y={sh - smb + 13}
                  textAnchor="middle" fontSize="8" fill="var(--text-dim)" fontWeight="600">{s.id}</text>
              </svg>

              {activeTHalf != null && (
                <div style={{ fontSize: 7, color: 'var(--text-dim)', textAlign: 'center' }}>
                  {`t½=${activeTHalf.toFixed(1)}h · n=${activeN ?? '?'} · R²=${activeR2?.toFixed(3) ?? '–'}`}
                  {fit && <span style={{ color: 'var(--accent)', marginLeft: 3 }}>✓ manual</span>}
                </div>
              )}

              <div style={{ fontSize: 7, color: nSel >= 3 ? 'var(--text-dim)' : '#B23A2E', textAlign: 'center' }}>
                {nSel} pt{nSel !== 1 ? 's' : ''} selected{nSel < 3 ? ' (≥3 req.)' : ''}
              </div>

              {errors[s.id] && (
                <div style={{ fontSize: 7, color: '#B23A2E', textAlign: 'center', maxWidth: sw, wordBreak: 'break-word' }}>
                  {errors[s.id]}
                </div>
              )}

              <div style={{ display: 'flex', gap: 3 }}>
                <button onClick={() => doRefit(s)} disabled={!canRefit || isLoading}
                  style={{
                    fontSize: 9, padding: '2px 8px', borderRadius: 8,
                    cursor: canRefit && !isLoading ? 'pointer' : 'default',
                    border: '1px solid var(--border)',
                    background: canRefit ? 'var(--accent)' : 'transparent',
                    color: canRefit ? '#fff' : 'var(--text-dim)',
                    opacity: isLoading ? 0.5 : 1,
                  }}>
                  {isLoading ? '…' : 'Refit λz'}
                </button>
                <button onClick={() => reset(s)}
                  style={{
                    fontSize: 9, padding: '2px 6px', borderRadius: 8, cursor: 'pointer',
                    border: '1px solid var(--border)', background: 'transparent', color: 'var(--text-dim)',
                  }}>
                  Reset
                </button>
              </div>
            </div>
          );
        })}
      </div>
      <div style={{ display: 'flex', gap: 12, fontSize: 10, color: 'var(--text-dim)', marginTop: 8 }}>
        <span><span style={{ color: 'var(--accent)' }}>●</span> Selected for λz</span>
        <span><span style={{ color: 'var(--text-dim)', opacity: 0.45 }}>●</span> Excluded</span>
        <span><span style={{ color: '#9A5B12' }}>– –</span> Regression fit</span>
      </div>
    </div>
  );
}

/** Inline-SVG prediction-corrected VPC panel. Shared by the main VPC card and
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
  );
}

const _VPC_STRUCTURAL_ROLES = new Set(
  ['ID', 'TIME', 'TAD', 'DV', 'AMT', 'EVID', 'MDV', 'CMT', 'II', 'ADDL', 'DVID', 'CENS', 'ROUTE', 'PD']);

/** Covariate columns eligible for VPC stratification: dataset columns without a
 * structural NONMEM role, plus DOSE (which the backend always accepts). The
 * backend's `available` list is authoritative — this is a best-effort menu. */
function vpcStrataOptions(
  meta: { columns?: { name: string }[]; detected_roles?: Record<string, string> } | null | undefined,
): string[] {
  const cols = meta?.columns ?? [];
  const roles = meta?.detected_roles ?? {};
  const covs = cols.map(c => c.name)
    .filter(n => !_VPC_STRUCTURAL_ROLES.has((roles[n] ?? '').toUpperCase()));
  return Array.from(new Set(['DOSE', ...covs]));
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
              {label} = {st.label} <span style={{ opacity: 0.7 }}>(n = {st.n})</span>
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
        {gb} = {g.label} <span style={{ opacity: 0.7 }}>(n = {g.n})</span>{' '}
        <span style={{ color: m.within ? 'var(--green)' : 'var(--red, #c0392b)' }}>
          {m.within ? '✓ within' : '✗ outside'}</span>
      </div>
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
      <svg viewBox={`0 0 ${W} ${H}`} width="48%" style={{ maxWidth: W }} role="img" aria-label="Observed vs predicted">
        <line x1={sx(0)} y1={sy(0)} x2={sx(hi)} y2={sy(hi)} stroke="var(--text-dim)" strokeDasharray="3 3" />
        <line x1={m} y1={H - m} x2={W - 8} y2={H - m} stroke="var(--border)" />
        <line x1={m} y1={8} x2={m} y2={H - m} stroke="var(--border)" />
        {obs.map((o, i) => <circle key={i} cx={sx(o)} cy={sy(ip[i])} r="2.2" fill="var(--accent)" fillOpacity="0.6" />)}
        <text x={(m + W) / 2} y={H - 6} textAnchor="middle" fontSize="10" fill="var(--text-dim)">observed</text>
        <text x={11} y={(8 + H - m) / 2} textAnchor="middle" fontSize="10" fill="var(--text-dim)"
          transform={`rotate(-90 11 ${(8 + H - m) / 2})`}>predicted (IPRED)</text>
      </svg>
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
      <svg viewBox={`0 0 ${W} ${H}`} width="48%" style={{ maxWidth: W }} role="img" aria-label="Visual predictive check">
        <polygon points={`${up} ${dn}`} fill="var(--accent)" fillOpacity="0.16" />
        <path d={med} fill="none" stroke="var(--accent)" strokeWidth="1.4" />
        {(r.obs_t ?? []).map((x, i) => <circle key={i} cx={sx(x)} cy={sy((r.obs_c ?? [])[i])} r="1.8" fill="var(--green)" fillOpacity="0.65" />)}
        <line x1={m} y1={H - m} x2={W - 8} y2={H - m} stroke="var(--border)" />
        <line x1={m} y1={8} x2={m} y2={H - m} stroke="var(--border)" />
        <text x={(m + W) / 2} y={H - 6} textAnchor="middle" fontSize="10" fill="var(--text-dim)">time (h)</text>
        <text x={11} y={(8 + H - m) / 2} textAnchor="middle" fontSize="10" fill="var(--text-dim)"
          transform={`rotate(-90 11 ${(8 + H - m) / 2})`}>conc.</text>
      </svg>
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
            <button key={m} className="chip" style={{
              padding: '1px 8px', marginLeft: 4,
              background: showMetric === m ? 'var(--accent)' : undefined,
              color: showMetric === m ? '#fff' : undefined,
            }} onClick={() => setShowMetric(m)}>{m === 'rel_auc' ? 'AUC' : 'Cmax'}</button>
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

const ROLE_OPTIONS = ['', 'ID', 'TIME', 'TAD', 'DV', 'AMT', 'EVID', 'MDV', 'CMT',
  'II', 'ADDL', 'DVID', 'CENS', 'ROUTE', 'PD'];

type ColMeta = { name: string; dtype: string; role: string };

function RolesEditor({ state, onApply, loading }:
  { state: PharmState; onApply: (o: Record<string, string>) => void; loading: boolean }) {
  const meta = state.dataset_metadata as { columns?: ColMeta[]; detected_roles?: Record<string, string> } | null;
  const cols = meta?.columns ?? [];
  const roles = meta?.detected_roles ?? {};
  const [edits, setEdits] = useState<Record<string, string>>({});
  if (cols.length === 0) return null;
  const roleFor = (c: string) => edits[c] ?? roles[c] ?? '';
  return (
    <div style={{ maxWidth: 560 }}>
      <div style={{ fontSize: 12, color: 'var(--text-dim)', marginBottom: 6 }}>
        Column roles — override any auto-detected mapping, then apply.
      </div>
      <table className="nca-table">
        <thead><tr><th>Column</th><th>Type</th><th>Role</th></tr></thead>
        <tbody>
          {cols.map(c => (
            <tr key={c.name}>
              <td>{c.name}</td>
              <td style={{ color: 'var(--text-dim)' }}>{c.dtype}</td>
              <td>
                <select className="model-select" style={{ maxWidth: 140 }} value={roleFor(c.name)}
                  disabled={loading}
                  onChange={e => setEdits(p => ({ ...p, [c.name]: e.target.value }))}>
                  {ROLE_OPTIONS.map(r => <option key={r} value={r}>{r || '—'}</option>)}
                </select>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      <button className="chip" style={{ marginTop: 8 }} disabled={loading || Object.keys(edits).length === 0}
        onClick={() => onApply(edits)}>Apply roles</button>
    </div>
  );
}

function SimChart({ sim }: { sim: PharmState['simulation_results'] }) {
  if (!sim || sim.status !== 'ok' || !sim.times || !sim.cp) {
    return <div className="qc-card conditional"><div className="qc-title">Simulation — not run</div>
      <div style={{ fontSize: 12 }}>{sim?.message}</div></div>;
  }
  const [logY, setLogY] = useState(false);
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

function AuditPanel({ entries, verified }: { entries: AuditEntry[]; verified: boolean }) {
  const [open, setOpen] = useState(false);
  return (
    <div>
      <span className="audit-toggle" onClick={() => setOpen(o => !o)}>
        {open ? '▾' : '▸'} Audit trail ({entries.length} entries)
      </span>
      {verified && (
        <span className="audit-ok" style={{ marginLeft: 8 }}>
          <ShieldCheck size={10} style={{ display: 'inline', marginRight: 3 }} />verified
        </span>
      )}
      {open && (
        <div className="audit-list">
          {entries.map(e => (
            <div className="audit-row" key={e.index} title={e.reason ? `reason: ${e.reason}` : undefined}>
              <span className="audit-row-idx">#{e.index}</span>
              <span className="audit-row-agent">{e.agent}</span>
              <span className="audit-row-tool">{e.tool}</span>
              {e.actor && e.actor !== 'anonymous' && (
                <span className="audit-row-actor" style={{ fontSize: 10, color: 'var(--text-dim)' }}>
                  by {e.actor}
                </span>
              )}
              <span style={{ marginLeft: 'auto', fontSize: 10 }}>{e.entry_hash.slice(0, 12)}…</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

export default function App() {
  const [session, setSession] = useState<Session | null>(null);
  const [state, setState] = useState<PharmState | null>(null);
  const [wfStatus, setWfStatus] = useState<WorkflowStatus>('idle');
  const [activeWorkflow, setActiveWorkflow] = useState<WorkflowName>('nca_full');
  const [messages, setMessages] = useState<DisplayMsg[]>([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [file, setFile] = useState<File | null>(null);
  const [drag, setDrag] = useState(false);
  const [audit, setAudit] = useState<AuditEntry[]>([]);
  const [auditVerified, setAuditVerified] = useState(false);
  const [healthy, setHealthy] = useState<boolean | null>(null);
  const [currentStep, setCurrentStep] = useState(-1);
  const [pkModels, setPkModels] = useState<PkModelDef[]>([]);
  const [selectedModel, setSelectedModel] = useState('oral_1cmt');
  const [simDose, setSimDose] = useState(100);
  const [simTau, setSimTau] = useState(24);
  const [simNDoses, setSimNDoses] = useState(1);
  const [simTmax, setSimTmax] = useState<number | ''>('');
  const [sweepDoses, setSweepDoses] = useState('');
  const [errorModel, setErrorModel] = useState('proportional');
  const [jobNote, setJobNote] = useState('');
  const [fcDose, setFcDose] = useState(100);
  const [fcTau, setFcTau] = useState(24);
  const [fcLevels, setFcLevels] = useState('');
  const [fcTarget, setFcTarget] = useState('');
  const [fcMetric, setFcMetric] = useState('cmin');
  const [seN, setSeN] = useState(20);
  const [seObsT, setSeObsT] = useState('0.5,1,2,4,8,12,24');
  const [seDose, setSeDose] = useState(100);
  const [seNRep, setSeNRep] = useState(5);
  const [seShowConfirm, setSeShowConfirm] = useState(false);
  const [token, setTokenState] = useState(getToken());
  const [showRoles, setShowRoles] = useState(false);
  const [skills, setSkills] = useState<SkillDef[]>([]);
  const [showSkills, setShowSkills] = useState(false);
  const [showFlexplot, setShowFlexplot] = useState(false);
  const messagesEnd = useRef<HTMLDivElement>(null);
  const fileRef = useRef<HTMLInputElement>(null);
  const shownMarkers = useRef(new Set<string>());

  const scrollBottom = () => messagesEnd.current?.scrollIntoView({ behavior: 'smooth' });
  useEffect(() => { scrollBottom(); }, [messages]);

  useEffect(() => {
    api.health()
      .then(h => setHealthy(h.status === 'ok'))
      .catch(() => setHealthy(false));
    api.createSession()
      .then(s => setSession(s))
      .catch(console.error);
    api.listPkModels()
      .then(r => setPkModels(r.models))
      .catch(() => { /* library unavailable */ });
  }, []);

  // Accept an optional `id` (some call sites pass `id: ''` as a placeholder) but
  // always assign a fresh unique id here so React keys stay stable and distinct.
  const pushMsg = useCallback((m: Omit<DisplayMsg, 'id'> & { id?: string }) => {
    setMessages(prev => [...prev, { ...m, id: `${Date.now()}-${Math.random()}` }]);
  }, []);

  function extractMessages(raw: AgentMessage[], agent: string): DisplayMsg[] {
    const out: DisplayMsg[] = [];
    for (const m of raw) {
      if (typeof m.content === 'string') {
        if (m.content.trim()) out.push({ role: m.role, content: m.content, agent, id: '' });
      } else if (Array.isArray(m.content)) {
        for (const block of m.content as ContentBlock[]) {
          if (block.type === 'text' && block.text?.trim()) {
            out.push({ role: m.role, content: block.text, agent, id: '' });
          } else if (block.type === 'tool_use') {
            out.push({ role: 'assistant', content: `Running: ${block.name}`, agent, tool: block.name, id: '' });
          }
        }
      }
    }
    return out.map(m => ({ ...m, id: `${Date.now()}-${Math.random()}` }));
  }

  async function refreshAudit() {
    if (!session) return;
    try {
      const a = await api.getAudit(session.id);
      setAudit(a.entries);
      setAuditVerified(a.verified);
    } catch { /* best-effort */ }
  }

  function handleFiles(f: File) {
    setFile(f);
  }
  function onFileChange(e: ChangeEvent<HTMLInputElement>) {
    if (e.target.files?.[0]) handleFiles(e.target.files[0]);
  }
  function onDrop(e: DragEvent) {
    e.preventDefault();
    setDrag(false);
    if (e.dataTransfer.files[0]) handleFiles(e.dataTransfer.files[0]);
  }

  function handleWorkflowResponse(res: { status: string; state: PharmState; messages?: AgentMessage[]; audit_ok: boolean }) {
    setState(res.state);
    setCurrentStep(res.state.current_step ?? -1);

    if (res.messages) {
      const agent = res.state.last_agent ?? 'supervisor';
      extractMessages(res.messages, agent).forEach(m => pushMsg(m));
    }

    if (res.state.spaghetti_data && !shownMarkers.current.has('spaghetti')) {
      shownMarkers.current.add('spaghetti');
      pushMsg({ role: 'assistant', content: '__SPAGHETTI__', agent: 'data_manager', id: '', snap: res.state });
    }
    if (res.state.nca_summary) {
      pushMsg({ role: 'assistant', content: '__NCA_TABLE__', agent: 'nca', id: '', snap: res.state });
    }
    if (res.state.nca_plot_data && !shownMarkers.current.has('nca_lz')) {
      shownMarkers.current.add('nca_lz');
      pushMsg({ role: 'assistant', content: '__NCA_LZ__', agent: 'nca', id: '', snap: res.state });
    }
    if (res.state.pk_model_results?.status === 'ok' && !shownMarkers.current.has('pkmodel')) {
      shownMarkers.current.add('pkmodel');
      pushMsg({ role: 'assistant', content: '__PKMODEL__', agent: 'modeler', id: '', snap: res.state });
    }
    if (res.state.engine_comparison_results && !shownMarkers.current.has('engines')) {
      shownMarkers.current.add('engines');
      pushMsg({ role: 'assistant', content: '__ENGINES__', agent: 'modeler', id: '', snap: res.state });
    }
    if (res.state.qc_verdict) {
      pushMsg({ role: 'assistant', content: '__QC_CARD__', agent: 'qc', id: '', snap: res.state });
    }

    if (res.status === 'awaiting_review') {
      setWfStatus('awaiting_review');
      refreshAudit();
    } else if (res.status === 'complete') {
      setWfStatus('complete');
      if (res.state.report_path) {
        pushMsg({ role: 'assistant', content: '__REPORT__', agent: 'report', id: '' });
      }
      refreshAudit();
    } else {
      setWfStatus('idle');
    }
  }

  async function uploadAndRun(workflow: WorkflowName = 'nca_full') {
    if (!session || !file) return;
    shownMarkers.current.clear();
    setActiveWorkflow(workflow);
    setLoading(true);
    setWfStatus('running');
    setCurrentStep(0);
    const wfLabel = WF_LABEL[workflow];
    pushMsg({ role: 'user', content: `Starting ${wfLabel} workflow on: ${file.name}`, id: '' });
    try {
      const up = await api.uploadDataset(session.id, file);
      const meta = up.metadata;
      pushMsg({
        role: 'assistant',
        content: `Dataset loaded: ${meta['n_records']} records, ${meta['n_subjects']} subjects, ${meta['n_columns']} columns.`,
        agent: 'data_manager',
        id: '',
      });
      const res = await api.startWorkflow(session.id, meta['dataset_path'] as string ?? '', workflow);
      handleWorkflowResponse(res);
    } catch (e) {
      pushMsg({ role: 'assistant', content: `Error: ${(e as Error).message}`, agent: 'supervisor', id: '' });
      setWfStatus('error');
    } finally {
      setLoading(false);
    }
  }

  async function resume(approve: boolean) {
    if (!session) return;
    setLoading(true);
    setWfStatus('running');
    // Approving runs every remaining step in one call. If any of them is a real
    // population fit, poll a job instead of holding the request open — and say
    // what actually happens next rather than assuming the NCA shape.
    const remaining = WORKFLOW_UI[activeWorkflow].steps.slice(Math.max(currentStep, 0));
    const isLongLeg = approve && remaining.some(s => HEAVY_STEPS.has(s.key));
    const note = !approve ? 'Rejected — workflow stopped.'
      : isLongLeg ? 'Approved — running the population fit (NLME → SCM → diagnostics → forest → VPC).'
      : 'Approved — generating report.';
    pushMsg({ role: 'user', content: note, id: '' });
    try {
      if (isLongLeg) {
        const { job_id } = await api.resumeWorkflowAsync(session.id);
        const res = await api.pollJob<WorkflowResponse>(session.id, job_id,
          s => setJobNote(`Population fit running… ${s}s (several real fits — this can take minutes)`));
        setJobNote('');
        handleWorkflowResponse(res);
        return;
      }
      const res = await api.resumeWorkflow(session.id, approve);
      handleWorkflowResponse(res);
    } catch (e) {
      pushMsg({ role: 'assistant', content: `Error: ${(e as Error).message}`, agent: 'supervisor', id: '' });
      setWfStatus('error');
    } finally {
      setLoading(false);
    }
  }

  const AGENT_CARD: Record<string, string> = {
    be: '__BE__', dose_prop: '__DP__',
    compartmental: '__COMPARTMENTAL__', poppk: '__POPPK__',
    nca: '__NCA_TABLE__', qc: '__QC_CARD__',
  };

  async function sendChat(preset?: string) {
    const text = (preset ?? input).trim();
    if (!session || !text) return;
    if (!preset) setInput('');
    pushMsg({ role: 'user', content: text, id: '' });
    setLoading(true);
    try {
      const res = await api.chat(session.id, text);
      setState(res.state);
      extractMessages(res.messages ?? [], res.agent).forEach(m => pushMsg(m));
      const marker = AGENT_CARD[res.agent];
      if (marker) pushMsg({ role: 'assistant', content: marker, agent: res.agent, id: '', snap: res.state });
    } catch (e) {
      pushMsg({ role: 'assistant', content: `Error: ${(e as Error).message}`, agent: 'supervisor', id: '' });
    } finally {
      setLoading(false);
    }
  }

  function onKeyDown(e: React.KeyboardEvent) {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendChat(); }
  }

  async function runPkModel(body: { model_key?: string; compare?: boolean }) {
    if (!session) return;
    setLoading(true);
    const label = body.compare ? 'Compare PK models'
      : `Fit ${pkModels.find(m => m.key === body.model_key)?.label ?? body.model_key}`;
    pushMsg({ role: 'user', content: label, id: '' });
    try {
      const res = await api.runPkModel(session.id, body);
      setState(res.state);
      pushMsg({ role: 'assistant', content: res.summary, agent: 'modeler', id: '' });
      pushMsg({ role: 'assistant', content: '__PKMODEL__', agent: 'modeler', id: '', snap: res.state });
    } catch (e) {
      pushMsg({ role: 'assistant', content: `Error: ${(e as Error).message}`, agent: 'modeler', id: '' });
    } finally {
      setLoading(false);
    }
  }

  async function runSimulate() {
    if (!session) return;
    setLoading(true);
    pushMsg({ role: 'user', content: `Simulate ${simNDoses}×${simDose} q${simTau}h`, id: '' });
    try {
      const res = await api.simulate(session.id, {
        dose: simDose, tau: simTau, n_doses: simNDoses,
        ...(simTmax ? { tmax: simTmax } : {}),
      });
      setState(res.state);
      pushMsg({ role: 'assistant', content: res.summary, agent: 'simulator', id: '' });
      pushMsg({ role: 'assistant', content: '__SIM__', agent: 'simulator', id: '', snap: res.state });
    } catch (e) {
      pushMsg({ role: 'assistant', content: `Error: ${(e as Error).message}`, agent: 'simulator', id: '' });
    } finally {
      setLoading(false);
    }
  }

  async function applyRoles(overrides: Record<string, string>) {
    if (!session) return;
    setLoading(true);
    try {
      const res = await api.setRoles(session.id, overrides);
      setState(res.state);
      setShowRoles(false);
      pushMsg({ role: 'assistant', content: 'Column roles updated.', agent: 'data_manager', id: '' });
    } catch (e) {
      pushMsg({ role: 'assistant', content: `Error: ${(e as Error).message}`, agent: 'data_manager', id: '' });
    } finally { setLoading(false); }
  }

  async function runVpc(opts?: { stratify_by?: string | null; dose_normalize?: boolean; x_by?: string;
    exposure_check?: boolean; blq_check?: boolean }) {
    if (!session) return;
    setLoading(true);
    const label = opts?.exposure_check ? ' — exposure predictive check'
      : opts?.blq_check ? ' — BLQ-incidence VPC'
      : opts?.stratify_by ? ` — stratified by ${opts.stratify_by}`
      : opts?.dose_normalize ? ' — dose-normalized' : '';
    pushMsg({ role: 'user', content: `VPC / goodness-of-fit${label}`, id: '' });
    try {
      const res = await api.vpc(session.id, opts);
      setState(res.state);
      pushMsg({ role: 'assistant', content: res.summary, agent: 'modeler', id: '' });
      pushMsg({ role: 'assistant', content: '__VPC__', agent: 'modeler', id: '', snap: res.state });
    } catch (e) {
      pushMsg({ role: 'assistant', content: `Error: ${(e as Error).message}`, agent: 'modeler', id: '' });
    } finally { setLoading(false); }
  }

  async function runReview() {
    if (!session) return;
    setLoading(true);
    pushMsg({ role: 'user', content: 'Adversarial review', id: '' });
    try {
      const res = await api.review(session.id);
      setState(res.state);
      const c = res.counts;
      const verdict = res.goal_met ? 'goal met' : 'findings block the goal';
      pushMsg({
        role: 'assistant',
        content: `Adversarial review (${res.iterations} pass${res.iterations === 1 ? '' : 'es'}): `
          + `${verdict} — ${c.CRITICAL} critical, ${c.HIGH} high, ${c.MEDIUM} medium, ${c.LOW} low.`,
        agent: 'reviewer', id: '',
      });
      pushMsg({ role: 'assistant', content: '__REVIEW__', agent: 'reviewer', id: '', snap: res.state });
    } catch (e) {
      pushMsg({ role: 'assistant', content: `Error: ${(e as Error).message}`, agent: 'reviewer', id: '' });
    } finally { setLoading(false); }
  }

  async function captureSkill() {
    if (!session) return;
    const name = window.prompt('Name this skill (reusable on new datasets):');
    if (!name) return;
    setLoading(true);
    try {
      const res = await api.captureSkill(session.id, { name: name.trim() });
      const tools = res.skill.steps.map(s => s.tool).join(' → ');
      pushMsg({
        role: 'assistant', agent: 'reviewer', id: '',
        content: `Captured skill **${res.skill.name}** (v${res.skill.version}): ${tools}. `
          + 'Replay it on a new dataset from the Skills panel.',
      });
      await refreshSkills();
    } catch (e) {
      pushMsg({ role: 'assistant', content: `Error: ${(e as Error).message}`, agent: 'reviewer', id: '' });
    } finally { setLoading(false); }
  }

  async function refreshSkills() {
    try {
      const res = await api.listSkills();
      setSkills(res.skills);
    } catch { /* skills are optional UI; ignore listing errors */ }
  }

  async function runSkill(name: string) {
    if (!state?.dataset_path) {
      pushMsg({ role: 'assistant', content: 'Load a dataset first to replay a skill.', agent: 'reviewer', id: '' });
      return;
    }
    setLoading(true);
    pushMsg({ role: 'user', content: `Replay skill: ${name}`, id: '' });
    try {
      const res = await api.runSkill(name, state.dataset_path);
      const ok = res.executed.filter(s => s.status === 'ok').length;
      const nca = res.state.nca_parameters?.length ?? 0;
      const rev = res.state.review_results;
      const revNote = rev ? ` Review ${rev.goal_met ? 'goal met' : `${rev.counts.CRITICAL + rev.counts.HIGH} blocker(s)`}.` : '';
      pushMsg({
        role: 'assistant', agent: 'reviewer', id: '',
        content: `Replayed **${name}** on the current dataset → new session \`${res.session_id}\` `
          + `(${ok}/${res.executed.length} steps ok${nca ? `, NCA n=${nca}` : ''}).${revNote}`,
      });
    } catch (e) {
      pushMsg({ role: 'assistant', content: `Error: ${(e as Error).message}`, agent: 'reviewer', id: '' });
    } finally { setLoading(false); }
  }

  async function deleteSkill(name: string) {
    if (!window.confirm(`Delete skill "${name}"?`)) return;
    try {
      await api.deleteSkill(name);
      await refreshSkills();
    } catch (e) {
      pushMsg({ role: 'assistant', content: `Error: ${(e as Error).message}`, agent: 'reviewer', id: '' });
    }
  }

  async function runNlme(method: string) {
    if (!session) return;
    const label: Record<string, string> = {
      focei: 'FOCE-I only', saem: 'SAEM',
      focei_saem: 'FOCE-I (SAEM-seeded)', auto: 'Auto (escalating)',
    };
    setLoading(true);
    pushMsg({ role: 'user', content: `NLME fit — ${label[method] ?? method} (${errorModel} error)`, id: '' });
    try {
      const { job_id } = await api.nlme(session.id, { method, error_model: errorModel });
      const res = await api.pollJob(session.id, job_id,
        s => setJobNote(`Population fit running… ${s}s`));
      setJobNote('');
      setState(res.state);
      pushMsg({ role: 'assistant', content: res.summary, agent: 'modeler', id: '' });
      pushMsg({ role: 'assistant', content: '__NLME__', agent: 'modeler', id: '', snap: res.state });
    } catch (e) {
      pushMsg({ role: 'assistant', content: `Error: ${(e as Error).message}`, agent: 'modeler', id: '' });
    } finally { setJobNote(''); setLoading(false); }
  }

  async function runScm() {
    if (!session) return;
    setLoading(true);
    pushMsg({ role: 'user', content: `Covariate search (SCM, ${errorModel} error)`, id: '' });
    try {
      const { job_id } = await api.scm(session.id, { error_model: errorModel });
      const res = await api.pollJob(session.id, job_id,
        s => setJobNote(`Covariate search running… ${s}s (this can take a few minutes)`));
      setJobNote('');
      setState(res.state);
      pushMsg({ role: 'assistant', content: res.summary, agent: 'modeler', id: '' });
      pushMsg({ role: 'assistant', content: '__SCM__', agent: 'modeler', id: '', snap: res.state });
    } catch (e) {
      pushMsg({ role: 'assistant', content: `Error: ${(e as Error).message}`, agent: 'modeler', id: '' });
    } finally { setJobNote(''); setLoading(false); }
  }

  async function runSimest() {
    if (!session) return;
    const obs_t = seObsT.split(',').map(s => Number(s.trim())).filter(Number.isFinite);
    if (!obs_t.length || seN < 2) return;
    setSeShowConfirm(false);
    setLoading(true);
    pushMsg({ role: 'user', content: `Simulation-estimation design check — N=${seN}, ${seNRep} replicate(s)`, id: '' });
    try {
      const { job_id } = await api.simest(session.id, {
        confirm: true,
        design: { n_subjects: seN, obs_t, dose: seDose, n_doses: 1 },
        n_rep: seNRep,
      });
      const res = await api.pollJob(session.id, job_id,
        s => setJobNote(`Simulation-estimation running… ${s}s (several real fits — this can take minutes)`));
      setJobNote('');
      setState(res.state);
      pushMsg({ role: 'assistant', content: res.summary, agent: 'simulator', id: '' });
      pushMsg({ role: 'assistant', content: '__SIMEST__', agent: 'simulator', id: '', snap: res.state });
    } catch (e) {
      pushMsg({ role: 'assistant', content: `Error: ${(e as Error).message}`, agent: 'simulator', id: '' });
    } finally { setJobNote(''); setLoading(false); }
  }

  async function runEngineComparison() {
    if (!session) return;
    setLoading(true);
    pushMsg({ role: 'user', content: 'Cross-engine comparison (FOCE-I vs nlmixr2)', id: '' });
    try {
      const { job_id } = await api.engineComparison(session.id,
        { engines: ['pharmagent_focei', 'nlmixr2'] });
      const res = await api.pollJob(session.id, job_id,
        s => setJobNote(`Cross-engine comparison running… ${s}s (fits + external engine can take a minute)`));
      setJobNote('');
      setState(res.state);
      pushMsg({ role: 'assistant', content: res.summary, agent: 'modeler', id: '' });
      pushMsg({ role: 'assistant', content: '__ENGINES__', agent: 'modeler', id: '', snap: res.state });
    } catch (e) {
      pushMsg({ role: 'assistant', content: `Error: ${(e as Error).message}`, agent: 'modeler', id: '' });
    } finally { setJobNote(''); setLoading(false); }
  }

  async function runForecast() {
    if (!session) return;
    const measured = fcLevels.split(/[;\n]+/).map(s => s.trim()).filter(Boolean)
      .map(pair => {
        const [t, c] = pair.split(/[,\s]+/).map(Number);
        return { time: t, conc: c };
      })
      .filter(m => Number.isFinite(m.time) && Number.isFinite(m.conc));
    setLoading(true);
    pushMsg({ role: 'user', content: `MAP forecast — ${fcDose} q${fcTau}h, ${measured.length} level(s)`, id: '' });
    try {
      const body: Parameters<typeof api.forecast>[1] = { dose: fcDose, tau: fcTau, measured };
      const tgt = Number(fcTarget);
      if (fcTarget.trim() !== '' && Number.isFinite(tgt)) { body.target = tgt; body.target_metric = fcMetric; }
      const res = await api.forecast(session.id, body);
      setState(res.state);
      pushMsg({ role: 'assistant', content: res.summary, agent: 'modeler', id: '' });
      pushMsg({ role: 'assistant', content: '__FORECAST__', agent: 'modeler', id: '', snap: res.state });
    } catch (e) {
      pushMsg({ role: 'assistant', content: `Error: ${(e as Error).message}`, agent: 'modeler', id: '' });
    } finally { setLoading(false); }
  }

  async function runDiagnostics() {
    if (!session) return;
    setLoading(true);
    pushMsg({ role: 'user', content: 'Residual diagnostics (IWRES / NPDE)', id: '' });
    try {
      const res = await api.diagnostics(session.id);
      setState(res.state);
      pushMsg({ role: 'assistant', content: res.summary, agent: 'modeler', id: '' });
      pushMsg({ role: 'assistant', content: '__DIAG__', agent: 'modeler', id: '', snap: res.state });
    } catch (e) {
      pushMsg({ role: 'assistant', content: `Error: ${(e as Error).message}`, agent: 'modeler', id: '' });
    } finally { setLoading(false); }
  }

  async function runCovariateForest() {
    if (!session) return;
    setLoading(true);
    pushMsg({ role: 'user', content: 'Covariate forest plot', id: '' });
    try {
      const res = await api.forest(session.id);
      setState(res.state);
      pushMsg({ role: 'assistant', content: res.summary, agent: 'modeler', id: '' });
      pushMsg({ role: 'assistant', content: '__FOREST__', agent: 'modeler', id: '', snap: res.state });
    } catch (e) {
      pushMsg({ role: 'assistant', content: `Error: ${(e as Error).message}`, agent: 'modeler', id: '' });
    } finally { setLoading(false); }
  }

  async function downloadFullReport() {
    if (!session) return;
    setLoading(true);
    try {
      const res = await api.generateReport(session.id);
      await api.downloadReportFile(session.id, res.result.report_path);
    } catch (e) {
      pushMsg({ role: 'assistant', content: `Error: ${(e as Error).message}`, agent: 'report', id: '' });
    } finally { setLoading(false); }
  }

  async function exportCsv(kind: string) {
    if (!session) return;
    try {
      await api.exportCsv(session.id, kind);
    } catch (e) {
      pushMsg({ role: 'assistant', content: `Error: ${(e as Error).message}`, agent: 'data_manager', id: '' });
    }
  }

  async function exportCdisc() {
    if (!session) return;
    try {
      await api.downloadCdisc(session.id);
    } catch (e) {
      pushMsg({ role: 'assistant', content: `Error: ${(e as Error).message}`, agent: 'data_manager', id: '' });
    }
  }

  async function exportControl(kind: 'nonmem' | 'mrgsolve') {
    if (!session) return;
    try {
      await api.exportControl(session.id, kind);
    } catch (e) {
      pushMsg({ role: 'assistant', content: `Error: ${(e as Error).message}`, agent: 'modeler', id: '' });
    }
  }

  async function runDoseSweep() {
    if (!session) return;
    setLoading(true);
    const doses = sweepDoses.split(',').map(s => Number(s.trim())).filter(n => n > 0);
    pushMsg({ role: 'user', content: `Dose sweep: ${doses.join(', ')}`, id: '' });
    try {
      const res = await api.doseSweep(session.id, {
        doses: doses.length ? doses : undefined, tau: simTau, n_doses: simNDoses,
        ...(simTmax ? { tmax: simTmax } : {}),
      });
      setState(res.state);
      pushMsg({ role: 'assistant', content: res.summary, agent: 'simulator', id: '' });
      pushMsg({ role: 'assistant', content: '__SWEEP__', agent: 'simulator', id: '', snap: res.state });
    } catch (e) {
      pushMsg({ role: 'assistant', content: `Error: ${(e as Error).message}`, agent: 'simulator', id: '' });
    } finally { setLoading(false); }
  }

  async function runClinsim(opts?: { doses?: number[]; metric?: string; threshold?: number | null;
    direction?: string; target_fraction?: number; n_subjects?: number; param_uncertainty?: boolean }) {
    if (!session) return;
    setLoading(true);
    pushMsg({ role: 'user', content: 'Clinical trial simulation (target attainment)', id: '' });
    try {
      const res = await api.clinsim(session.id, {
        ...(opts?.doses?.length ? { doses: opts.doses } : { dose: simDose }),
        tau: simTau, n_doses: simNDoses,
        ...(opts?.metric ? { metric: opts.metric } : {}),
        ...(opts && 'threshold' in opts ? { threshold: opts.threshold } : {}),
        ...(opts?.direction ? { direction: opts.direction } : {}),
        ...(opts?.target_fraction != null ? { target_fraction: opts.target_fraction } : {}),
        ...(opts?.n_subjects ? { n_subjects: opts.n_subjects } : {}),
        ...(opts?.param_uncertainty ? { param_uncertainty: true } : {}),
      });
      setState(res.state);
      pushMsg({ role: 'assistant', content: res.summary, agent: 'simulator', id: '' });
      pushMsg({ role: 'assistant', content: '__CLINSIM__', agent: 'simulator', id: '', snap: res.state });
    } catch (e) {
      pushMsg({ role: 'assistant', content: `Error: ${(e as Error).message}`, agent: 'simulator', id: '' });
    } finally { setLoading(false); }
  }

  async function runExposureForest() {
    if (!session) return;
    setLoading(true);
    pushMsg({ role: 'user', content: 'Exposure covariate forest', id: '' });
    try {
      const res = await api.exposureForest(session.id, { dose: simDose, tau: simTau, n_doses: simNDoses });
      setState(res.state);
      pushMsg({ role: 'assistant', content: res.summary, agent: 'simulator', id: '' });
      pushMsg({ role: 'assistant', content: '__EXPFOREST__', agent: 'simulator', id: '', snap: res.state });
    } catch (e) {
      pushMsg({ role: 'assistant', content: `Error: ${(e as Error).message}`, agent: 'simulator', id: '' });
    } finally { setLoading(false); }
  }

  const QUICK_ACTIONS = [
    { label: 'Dose proportionality', msg: 'run dose proportionality power model' },
    { label: 'Compartmental fit', msg: 'fit a one- and two-compartment model' },
    { label: 'Population PK', msg: 'population pk typical values and iiv' },
    { label: 'Bioequivalence', msg: 'run a bioequivalence assessment test vs reference' },
  ];
  const hasData = !!state?.dataset_id;
  const EXPORT_LABEL: Record<string, string> = {
    nca: 'NCA', be: 'BE', dose_prop: 'Dose-prop', pk_model: 'Model',
    nlme: 'NLME', dose_sweep: 'Dose sweep', vpc: 'VPC',
  };
  const exportKinds = (state ? [
    state.nca_parameters?.length ? 'nca' : null,
    state.be_results?.status === 'ok' ? 'be' : null,
    state.dose_prop_results?.status === 'ok' ? 'dose_prop' : null,
    state.pk_model_results?.status === 'ok' ? 'pk_model' : null,
    state.nlme_results?.status === 'ok' ? 'nlme' : null,
    state.dose_sweep_results?.status === 'ok' ? 'dose_sweep' : null,
    state.vpc_results?.status === 'ok' ? 'vpc' : null,
  ] : []).filter(Boolean) as string[];

  const canRunWorkflow = !!session && !!file && wfStatus === 'idle' && !loading;

  return (
    <>
      <header className="topbar">
        <div className="topbar-logo">
          <FlaskConical size={18} />
          PharmAgent
          <span className="topbar-badge">PmatricsAI</span>
        </div>
        <div className="topbar-right">
          <input
            className="token-input"
            type="password"
            placeholder="API token (optional)"
            value={token}
            onChange={e => { setTokenState(e.target.value); setToken(e.target.value); }}
            title="Bearer token — required only when the backend has PHARMAGENT_API_TOKEN set"
          />
          <div className="status-dot" style={{ background: healthy === false ? 'var(--red)' : 'var(--green)' }} />
          <span className="status-text">
            {healthy === null ? 'connecting…' : healthy ? 'Backend online · MockLLM' : 'Backend offline'}
          </span>
        </div>
      </header>

      <aside className="sidebar">
        <div className="sidebar-section">
          <div className="sidebar-label">Session</div>
          <div className="sidebar-stat">
            <span className="sidebar-stat-key">ID</span>
            <span className="sidebar-stat-val" style={{ fontSize: 11, fontFamily: 'var(--mono)' }}>
              {session ? session.id.slice(0, 16) + '…' : '–'}
            </span>
          </div>
          <div className="sidebar-stat">
            <span className="sidebar-stat-key">Subjects</span>
            <span className="sidebar-stat-val">
              {state?.dataset_metadata ? String(state.dataset_metadata['n_subjects'] ?? '–') : '–'}
            </span>
          </div>
          <div className="sidebar-stat">
            <span className="sidebar-stat-key">QC</span>
            <span className="sidebar-stat-val" style={{
              color: state?.qc_verdict === 'PASS' ? 'var(--green)'
                : state?.qc_verdict ? 'var(--yellow)' : 'var(--text-dim)',
            }}>
              {state?.qc_verdict ?? '–'}
            </span>
          </div>
          <div className="sidebar-stat">
            <span className="sidebar-stat-key">Audit</span>
            <span className="sidebar-stat-val" style={{ color: auditVerified ? 'var(--green)' : 'var(--text-dim)' }}>
              {audit.length > 0 ? `${audit.length} entries${auditVerified ? ' ✓' : ''}` : '–'}
            </span>
          </div>
        </div>

        <div className="sidebar-section">
          <div className="sidebar-label">{WORKFLOW_UI[activeWorkflow].title}</div>
          <ul className="step-list">
            {WORKFLOW_UI[activeWorkflow].steps.map((s, i) => {
              const done = currentStep > i || wfStatus === 'complete';
              const active = currentStep === i && wfStatus === 'running';
              const gate = 'gate' in s && s.gate && wfStatus === 'awaiting_review';
              const cls = gate ? 'gate' : done ? 'done' : active ? 'active' : 'pending';
              return (
                <li key={s.key} className={`step-item ${cls}`}>
                  <span className="step-dot" />
                  {s.label}
                  {active && <Loader2 size={10} style={{ marginLeft: 'auto', animation: 'spin 1s linear infinite' }} />}
                  {gate && <AlertTriangle size={10} style={{ marginLeft: 'auto' }} />}
                </li>
              );
            })}
          </ul>
        </div>

        <div className="sidebar-section" style={{ marginTop: 'auto' }}>
          <button className="workflow-btn" disabled={!canRunWorkflow} onClick={() => uploadAndRun('nca_full')}>
            {loading && wfStatus === 'running' && activeWorkflow === 'nca_full'
              ? <><div className="spinner" /> Running…</>
              : <><Activity size={13} /> Run NCA Workflow</>}
          </button>
          <button className="workflow-btn" disabled={!canRunWorkflow} onClick={() => uploadAndRun('poppk_modeling')}
            style={{ marginTop: 8 }}
            title="Fit structural models, then confirm across estimation engines (native FOCE-I + nlmixr2), reviewed and QC-gated">
            {loading && wfStatus === 'running' && activeWorkflow === 'poppk_modeling'
              ? <><div className="spinner" /> Running…</>
              : <><Activity size={13} /> Run Modeling + Engines</>}
          </button>
          <button className="workflow-btn" disabled={!canRunWorkflow} onClick={() => uploadAndRun('poppk_full')}
            style={{ marginTop: 8 }}
            title="Full population PK: structural comparison (gated), NLME fit, SCM covariate build, residual diagnostics, covariate forest, VPC, adversarial review (gated), report">
            {loading && wfStatus === 'running' && activeWorkflow === 'poppk_full'
              ? <><div className="spinner" /> Running…</>
              : <><Activity size={13} /> Run Full PopPK</>}
          </button>
        </div>
      </aside>

      <main className="main">
        {wfStatus === 'idle' && (
          <div
            className={`upload-zone ${drag ? 'drag' : ''}`}
            onDragOver={e => { e.preventDefault(); setDrag(true); }}
            onDragLeave={() => setDrag(false)}
            onDrop={onDrop}
            onClick={() => fileRef.current?.click()}
          >
            <input ref={fileRef} type="file" accept=".csv" onChange={onFileChange} />
            <Upload size={22} style={{ color: 'var(--text-dim)', marginBottom: 8 }} />
            <div className="upload-label">
              <strong>Click to upload</strong> or drag & drop
            </div>
            <div className="upload-hint">CSV · NONMEM-style (ID / TIME / DV / AMT columns)</div>
            {file && (
              <div className="upload-file-name">
                <CheckCircle size={13} /> {file.name}
              </div>
            )}
          </div>
        )}

        {wfStatus === 'awaiting_review' && (
          <div className="gate-banner">
            <AlertTriangle size={20} style={{ color: 'var(--yellow)', flexShrink: 0 }} />
            <div className="gate-banner-text">
              <div className="gate-banner-title">Human Review Required</div>
              <div className="gate-banner-sub">QC complete — approve to generate DOCX report.</div>
            </div>
            <div className="gate-actions">
              <button className="btn btn-green" disabled={loading} onClick={() => resume(true)}>
                <CheckCircle size={12} /> Approve
              </button>
              <button className="btn btn-red" disabled={loading} onClick={() => resume(false)}>
                <XCircle size={12} /> Reject
              </button>
            </div>
          </div>
        )}

        {wfStatus === 'complete' && state?.report_path && (
          <div className="report-banner">
            <FileText size={20} style={{ color: 'var(--green)', flexShrink: 0 }} />
            <div className="report-banner-text">
              <div className="report-banner-title">Report ready</div>
              <div className="report-banner-sub">{state.report_path.split('/').pop()}</div>
            </div>
            <a className="btn btn-green" href={api.downloadReport(session!.id, state.report_path)} download>
              <Download size={12} /> Download DOCX
            </a>
          </div>
        )}

        <div className="messages">
          {messages.length === 0 && (
            <div className="empty">
              <FlaskConical size={36} />
              <span>Upload a PK dataset and run the NCA workflow</span>
              <span style={{ fontSize: 12 }}>Or type a message to chat with agents directly</span>
            </div>
          )}
          {messages.map(m => {
            // Read from the message's frozen snapshot when present (set at the
            // time the card was produced), else the live state.
            const st = m.snap ?? state;
            if (m.content === '__SPAGHETTI__' && st?.spaghetti_data) {
              return (
                <div key={m.id} className="msg agent">
                  <div className="msg-avatar" style={{ color: 'var(--agent-data)' }}>DM</div>
                  <div className="msg-bubble">
                    <div className="msg-agent-tag" style={{ color: 'var(--agent-data)' }}>Data Manager</div>
                    <div style={{ fontWeight: 600, fontSize: 12, marginBottom: 6 }}>Concentration–Time Plot</div>
                    <SpaghettiChart data={st.spaghetti_data as SpaghettiData} />
                  </div>
                </div>
              );
            }
            if (m.content === '__NCA_TABLE__' && st?.nca_summary) {
              return (
                <div key={m.id} className="msg agent">
                  <div className="msg-avatar" style={{ color: 'var(--agent-nca)' }}>NC</div>
                  <div className="msg-bubble">
                    <div className="msg-agent-tag" style={{ color: 'var(--agent-nca)' }}>NCA Agent</div>
                    <NcaTable state={st} />
                  </div>
                </div>
              );
            }
            if (m.content === '__NCA_LZ__' && st?.nca_plot_data) {
              return (
                <div key={m.id} className="msg agent">
                  <div className="msg-avatar" style={{ color: 'var(--agent-nca)' }}>NC</div>
                  <div className="msg-bubble">
                    <div className="msg-agent-tag" style={{ color: 'var(--agent-nca)' }}>NCA Agent</div>
                    <div style={{ fontWeight: 600, fontSize: 12, marginBottom: 6 }}>Terminal Slope (λz) Diagnostics</div>
                    <NcaLzPlot data={st.nca_plot_data as NcaPlotData} sessionId={session?.id ?? ''} />
                  </div>
                </div>
              );
            }
            if (m.content === '__QC_CARD__' && st?.qc_verdict) {
              return (
                <div key={m.id} className="msg agent">
                  <div className="msg-avatar" style={{ color: 'var(--agent-qc)' }}>QC</div>
                  <div className="msg-bubble">
                    <div className="msg-agent-tag" style={{ color: 'var(--agent-qc)' }}>QC Agent</div>
                    <QcCard state={st} />
                    {audit.length > 0 && (
                      <div style={{ marginTop: 10 }}>
                        <AuditPanel entries={audit} verified={auditVerified} />
                      </div>
                    )}
                  </div>
                </div>
              );
            }
            if (m.content === '__BE__' && st?.be_results) {
              return (
                <div key={m.id} className="msg agent">
                  <div className="msg-avatar" style={{ color: 'var(--agent-qc)' }}>BE</div>
                  <div className="msg-bubble">
                    <div className="msg-agent-tag" style={{ color: 'var(--agent-qc)' }}>Bioequivalence Agent</div>
                    <BeCard r={st.be_results} />
                  </div>
                </div>
              );
            }
            if (m.content === '__DP__' && st?.dose_prop_results) {
              return (
                <div key={m.id} className="msg agent">
                  <div className="msg-avatar" style={{ color: 'var(--agent-data)' }}>DP</div>
                  <div className="msg-bubble">
                    <div className="msg-agent-tag" style={{ color: 'var(--agent-data)' }}>Dose-Proportionality Agent</div>
                    <DosePropCard r={st.dose_prop_results} />
                  </div>
                </div>
              );
            }
            if (m.content === '__COMPARTMENTAL__' && st?.compartmental_results) {
              return (
                <div key={m.id} className="msg agent">
                  <div className="msg-avatar" style={{ color: 'var(--agent-nca)' }}>CM</div>
                  <div className="msg-bubble">
                    <div className="msg-agent-tag" style={{ color: 'var(--agent-nca)' }}>Compartmental Agent</div>
                    <CompartmentalCard r={st.compartmental_results} />
                  </div>
                </div>
              );
            }
            if (m.content === '__POPPK__' && st?.poppk_results) {
              return (
                <div key={m.id} className="msg agent">
                  <div className="msg-avatar" style={{ color: 'var(--agent-report)' }}>PK</div>
                  <div className="msg-bubble">
                    <div className="msg-agent-tag" style={{ color: 'var(--agent-report)' }}>Population PK Agent</div>
                    <PopPkCard r={st.poppk_results} />
                  </div>
                </div>
              );
            }
            if (m.content === '__PKMODEL__' && st?.pk_model_results) {
              return (
                <div key={m.id} className="msg agent">
                  <div className="msg-avatar" style={{ color: 'var(--agent-nca)' }}>MD</div>
                  <div className="msg-bubble">
                    <div className="msg-agent-tag" style={{ color: 'var(--agent-nca)' }}>Modeler Agent</div>
                    <PkModelCard r={st.pk_model_results} />
                  </div>
                </div>
              );
            }
            if (m.content === '__VPC__' && st?.vpc_results) {
              const wide = !!(st.vpc_results.stratified || st.vpc_results.exposure_pc
                || st.vpc_results.blq_vpc);
              return (
                <div key={m.id} className="msg agent">
                  <div className="msg-avatar" style={{ color: 'var(--agent-nca)' }}>VP</div>
                  <div className="msg-bubble" style={{ maxWidth: wide ? 920 : 640 }}>
                    <div className="msg-agent-tag" style={{ color: 'var(--agent-nca)' }}>Modeler · VPC / GOF</div>
                    <VpcCard r={st.vpc_results} onRerun={runVpc} busy={loading}
                      covariates={vpcStrataOptions(st.dataset_metadata as
                        { columns?: { name: string }[]; detected_roles?: Record<string, string> } | null)} />
                  </div>
                </div>
              );
            }
            if (m.content === '__NLME__' && st?.nlme_results) {
              return (
                <div key={m.id} className="msg agent">
                  <div className="msg-avatar" style={{ color: 'var(--agent-nca)' }}>NL</div>
                  <div className="msg-bubble" style={{ maxWidth: 640 }}>
                    <div className="msg-agent-tag" style={{ color: 'var(--agent-nca)' }}>Modeler · NLME (mixed-effects)</div>
                    <NlmeCard r={st.nlme_results} />
                  </div>
                </div>
              );
            }
            if (m.content === '__SCM__' && st?.scm_results) {
              return (
                <div key={m.id} className="msg agent">
                  <div className="msg-avatar" style={{ color: 'var(--agent-nca)' }}>CM</div>
                  <div className="msg-bubble" style={{ maxWidth: 680 }}>
                    <div className="msg-agent-tag" style={{ color: 'var(--agent-nca)' }}>Modeler · Covariate model (SCM)</div>
                    <ScmCard r={st.scm_results} />
                  </div>
                </div>
              );
            }
            if (m.content === '__ENGINES__' && st?.engine_comparison_results) {
              return (
                <div key={m.id} className="msg agent">
                  <div className="msg-avatar" style={{ color: 'var(--agent-nca)' }}>CE</div>
                  <div className="msg-bubble" style={{ maxWidth: 700 }}>
                    <div className="msg-agent-tag" style={{ color: 'var(--agent-nca)' }}>Modeler · Cross-engine comparison</div>
                    <EngineComparisonCard r={st.engine_comparison_results} />
                  </div>
                </div>
              );
            }
            if (m.content === '__FORECAST__' && st?.forecast_results) {
              return (
                <div key={m.id} className="msg agent">
                  <div className="msg-avatar" style={{ color: 'var(--agent-nca)' }}>FC</div>
                  <div className="msg-bubble" style={{ maxWidth: 560 }}>
                    <div className="msg-agent-tag" style={{ color: 'var(--agent-nca)' }}>Modeler · MAP / TDM forecast</div>
                    <ForecastCard r={st.forecast_results} />
                  </div>
                </div>
              );
            }
            if (m.content === '__DIAG__' && st?.diagnostics_results) {
              return (
                <div key={m.id} className="msg agent">
                  <div className="msg-avatar" style={{ color: 'var(--agent-nca)' }}>DG</div>
                  <div className="msg-bubble" style={{ maxWidth: 660 }}>
                    <div className="msg-agent-tag" style={{ color: 'var(--agent-nca)' }}>Modeler · Residual diagnostics</div>
                    <DiagnosticsCard r={st.diagnostics_results} />
                  </div>
                </div>
              );
            }
            if (m.content === '__FOREST__' && st?.forest_results) {
              return (
                <div key={m.id} className="msg agent">
                  <div className="msg-avatar" style={{ color: 'var(--agent-nca)' }}>CF</div>
                  <div className="msg-bubble" style={{ maxWidth: 660 }}>
                    <div className="msg-agent-tag" style={{ color: 'var(--agent-nca)' }}>Modeler · Covariate forest plot</div>
                    <ForestCard r={st.forest_results} />
                  </div>
                </div>
              );
            }
            if (m.content === '__SIMEST__' && st?.simest_results) {
              return (
                <div key={m.id} className="msg agent">
                  <div className="msg-avatar" style={{ color: 'var(--accent)' }}>SE</div>
                  <div className="msg-bubble" style={{ maxWidth: 660 }}>
                    <div className="msg-agent-tag" style={{ color: 'var(--accent)' }}>Simulator · Trial-design precision check</div>
                    <SimestCard r={st.simest_results} />
                  </div>
                </div>
              );
            }
            if (m.content === '__SWEEP__' && st?.dose_sweep_results) {
              return (
                <div key={m.id} className="msg agent">
                  <div className="msg-avatar" style={{ color: 'var(--accent)' }}>DS</div>
                  <div className="msg-bubble" style={{ maxWidth: 640 }}>
                    <div className="msg-agent-tag" style={{ color: 'var(--accent)' }}>Simulator · Dose sweep</div>
                    <DoseSweepCard r={st.dose_sweep_results} />
                  </div>
                </div>
              );
            }
            if (m.content === '__CLINSIM__' && st?.clinsim_results) {
              return (
                <div key={m.id} className="msg agent">
                  <div className="msg-avatar" style={{ color: 'var(--accent)' }}>CT</div>
                  <div className="msg-bubble" style={{ maxWidth: 660 }}>
                    <div className="msg-agent-tag" style={{ color: 'var(--accent)' }}>Simulator · Clinical trial simulation</div>
                    <ClinsimCard r={st.clinsim_results} onRerun={runClinsim} busy={loading} />
                  </div>
                </div>
              );
            }
            if (m.content === '__EXPFOREST__' && st?.exposure_forest_results) {
              return (
                <div key={m.id} className="msg agent">
                  <div className="msg-avatar" style={{ color: 'var(--accent)' }}>EF</div>
                  <div className="msg-bubble" style={{ maxWidth: 680 }}>
                    <div className="msg-agent-tag" style={{ color: 'var(--accent)' }}>Simulator · Exposure covariate forest</div>
                    <ExposureForestCard r={st.exposure_forest_results} />
                  </div>
                </div>
              );
            }
            if (m.content === '__SIM__' && st?.simulation_results) {
              return (
                <div key={m.id} className="msg agent">
                  <div className="msg-avatar" style={{ color: 'var(--accent)' }}>SM</div>
                  <div className="msg-bubble" style={{ maxWidth: 640 }}>
                    <div className="msg-agent-tag" style={{ color: 'var(--accent)' }}>Simulator</div>
                    <SimChart sim={st.simulation_results} />
                  </div>
                </div>
              );
            }
            if (m.content === '__REVIEW__' && st?.review_results) {
              return (
                <div key={m.id} className="msg agent">
                  <div className="msg-avatar" style={{ color: 'var(--agent-qc)' }}>AR</div>
                  <div className="msg-bubble" style={{ maxWidth: 720 }}>
                    <div className="msg-agent-tag" style={{ color: 'var(--agent-qc)' }}>Reviewer · Adversarial review</div>
                    <ReviewCard r={st.review_results} />
                  </div>
                </div>
              );
            }
            if (m.content === '__REPORT__') return null;
            return <MessageBubble key={m.id} msg={m} agent={m.agent} />;
          })}
          <div ref={messagesEnd} />
        </div>

        {hasData && (
          <div className="quick-actions" style={{ flexDirection: 'column', alignItems: 'flex-start' }}>
            <div>
              <span className="quick-actions-label">Data:</span>
              <button className="chip" disabled={loading} onClick={() => setShowRoles(s => !s)}>
                {showRoles ? 'Hide columns' : 'Columns / roles'}
              </button>
            </div>
            {showRoles && state && <RolesEditor state={state} onApply={applyRoles} loading={loading} />}
          </div>
        )}

        {hasData && (
          <div className="quick-actions" style={{ flexDirection: 'column', alignItems: 'stretch' }}>
            <div>
              <span className="quick-actions-label">Visualize:</span>
              <button className="chip" disabled={loading} onClick={() => setShowFlexplot(s => !s)}>
                {showFlexplot ? 'Hide flexplot' : 'Flexplot'}
              </button>
            </div>
            {showFlexplot && session && <FlexplotPanel sessionId={session.id} />}
          </div>
        )}

        {hasData && pkModels.length > 0 && (
          <div className="quick-actions">
            <span className="quick-actions-label">PK model library:</span>
            <select
              className="model-select"
              value={selectedModel}
              onChange={e => setSelectedModel(e.target.value)}
              disabled={loading}
            >
              {['IV linear', 'Oral', 'Nonlinear', 'PK/PD'].map(group => (
                <optgroup key={group} label={group}>
                  {pkModels.filter(m => m.group === group).map(m => (
                    <option key={m.key} value={m.key}>{m.label}{m.has_pd ? ' (needs PD)' : ''}</option>
                  ))}
                </optgroup>
              ))}
            </select>
            <button className="chip" disabled={loading} onClick={() => runPkModel({ model_key: selectedModel })}>
              Fit model
            </button>
            <button className="chip" disabled={loading} onClick={() => runPkModel({ compare: true })}>
              Compare oral models
            </button>
          </div>
        )}

        {state?.pk_model_results?.status === 'ok' && (
          <div className="quick-actions">
            <span className="quick-actions-label">Forecast:</span>
            <label className="sim-field">dose
              <input type="number" value={simDose} disabled={loading}
                onChange={e => setSimDose(Number(e.target.value))} />
            </label>
            <label className="sim-field">q (h)
              <input type="number" value={simTau} disabled={loading}
                onChange={e => setSimTau(Number(e.target.value))} />
            </label>
            <label className="sim-field"># doses
              <input type="number" value={simNDoses} disabled={loading}
                onChange={e => setSimNDoses(Number(e.target.value))} />
            </label>
            <label className="sim-field">to (h)
              <input type="number" placeholder="auto" value={simTmax} disabled={loading}
                onChange={e => setSimTmax(e.target.value === '' ? '' : Number(e.target.value))} />
            </label>
            <button className="chip" disabled={loading} onClick={runSimulate}>Simulate forward</button>
          </div>
        )}

        {state?.pk_model_results?.status === 'ok' && (
          <div className="quick-actions">
            <span className="quick-actions-label">Population (NLME):</span>
            <button className="chip" disabled={loading} onClick={() => runNlme('focei')}
              title="Single cold start — fastest and fully reproducible. On harder models (several IIV terms, covariates) a cold start can converge to the wrong optimum while still reporting success.">
              FOCE-I only
            </button>
            <button className="chip" disabled={loading} onClick={() => runNlme('saem')}
              title="Stochastic EM — explores rather than descends, so it is far less sensitive to starting values, but gives no exact Laplace OFV or asymptotic standard errors.">
              SAEM
            </button>
            <button className="chip" disabled={loading} onClick={() => runNlme('auto')}
              title="Runs FOCE-I, then probes with an independent SAEM-seeded start. If the two agree it stops there; if they disagree it escalates to a multi-start search and returns the lowest-OFV fit. Never worse than FOCE-I alone, but much slower whenever it escalates.">
              Auto
            </button>
            <label className="sim-field">error
              <select className="model-select" style={{ maxWidth: 130 }} value={errorModel}
                disabled={loading} onChange={e => setErrorModel(e.target.value)}>
                <option value="proportional">proportional</option>
                <option value="additive">additive</option>
                <option value="combined">combined</option>
              </select>
            </label>
            <button className="chip" disabled={loading} onClick={runScm}
              title="Stepwise covariate modeling: forward selection (p<0.05) + backward elimination (p<0.01) over dataset covariates">
              Covariate SCM
            </button>
            <button className="chip" disabled={loading} onClick={runEngineComparison}
              title="Fit the model across estimation engines (native FOCE-I + nlmixr2 if installed); winner chosen by prediction accuracy, not cross-engine OFV">
              Compare engines
            </button>
          </div>
        )}

        {state?.nlme_results?.status === 'ok' && (
          <div className="quick-actions">
            <span className="quick-actions-label">TDM / MAP forecast:</span>
            <label className="sim-field">dose
              <input type="number" value={fcDose} disabled={loading}
                onChange={e => setFcDose(Number(e.target.value))} />
            </label>
            <label className="sim-field">q (h)
              <input type="number" value={fcTau} disabled={loading}
                onChange={e => setFcTau(Number(e.target.value))} />
            </label>
            <label className="sim-field">levels (t,conc; …)
              <input type="text" style={{ width: 150 }} placeholder="e.g. 48.5,1.2; 72,0.4"
                value={fcLevels} disabled={loading} onChange={e => setFcLevels(e.target.value)} />
            </label>
            <label className="sim-field">target
              <input type="text" style={{ width: 60 }} placeholder="opt." value={fcTarget}
                disabled={loading} onChange={e => setFcTarget(e.target.value)} />
            </label>
            <select className="model-select" style={{ maxWidth: 90 }} value={fcMetric}
              disabled={loading} onChange={e => setFcMetric(e.target.value)}>
              <option value="cmin">Cmin</option>
              <option value="cmax">Cmax</option>
              <option value="cavg">Cavg</option>
              <option value="auc_tau">AUCτ</option>
            </select>
            <button className="chip" disabled={loading} onClick={runForecast}
              title="MAP/empirical-Bayes individualization from the fitted population model + measured levels">
              MAP forecast
            </button>
          </div>
        )}

        {state?.nlme_results?.status === 'ok' && !(state.nlme_results.covariate_effects?.length) && (
          <div className="quick-actions">
            <span className="quick-actions-label">Trial-design precision check:</span>
            {!seShowConfirm ? (
              <button className="chip" disabled={loading} onClick={() => setSeShowConfirm(true)}
                title="Simulate replicate trials under a proposed design and re-fit each — checks whether the 95% CI lands within 60-140% of its own estimate (up to 10 replicates; runs several real NLME fits, several minutes)">
                Simulation-estimation…
              </button>
            ) : (
              <>
                <label className="sim-field">N subjects
                  <input type="number" value={seN} disabled={loading}
                    onChange={e => setSeN(Number(e.target.value))} />
                </label>
                <label className="sim-field">sample times (h)
                  <input type="text" style={{ width: 160 }} value={seObsT} disabled={loading}
                    onChange={e => setSeObsT(e.target.value)} />
                </label>
                <label className="sim-field">dose
                  <input type="number" value={seDose} disabled={loading}
                    onChange={e => setSeDose(Number(e.target.value))} />
                </label>
                <label className="sim-field">replicates (≤10)
                  <input type="number" min={1} max={10} value={seNRep} disabled={loading}
                    onChange={e => setSeNRep(Math.max(1, Math.min(10, Number(e.target.value))))} />
                </label>
                <button className="chip" disabled={loading} onClick={runSimest}
                  title="Confirms and runs — several real NLME fits, several minutes to tens of minutes; holds this session while running">
                  Confirm &amp; run
                </button>
                <button className="chip" disabled={loading} onClick={() => setSeShowConfirm(false)}>
                  Cancel
                </button>
              </>
            )}
          </div>
        )}
        {state?.nlme_results?.status === 'ok' && !!(state.nlme_results.covariate_effects?.length) && (
          <div className="quick-actions">
            <span className="quick-actions-label" style={{ color: 'var(--text-dim)' }}>
              Trial-design precision check unavailable: not supported for models with covariate effects.
            </span>
          </div>
        )}

        {state?.pk_model_results?.status === 'ok' && (
          <div className="quick-actions">
            <span className="quick-actions-label">Diagnostics:</span>
            <button className="chip" disabled={loading} onClick={() => runVpc()}>VPC / goodness-of-fit</button>
            <button className="chip" disabled={loading} onClick={runDiagnostics}>Residual diagnostics</button>
            <button className="chip" disabled={loading} onClick={runCovariateForest}
              title="Covariate GMR forest plot from a converged run_nlme or run_scm covariate model">
              Covariate forest
            </button>
            <button className="chip" disabled={loading} onClick={runExposureForest}
              title="Simulated exposure forest: relative AUC/Cmax across covariate extremes with the 0.8–1.25 band">
              Exposure forest
            </button>
            <label className="sim-field">doses
              <input type="text" style={{ width: 130 }} placeholder="e.g. 2500,5000,10000"
                value={sweepDoses} disabled={loading}
                onChange={e => setSweepDoses(e.target.value)} />
            </label>
            <button className="chip" disabled={loading} onClick={runDoseSweep}>Dose sweep</button>
            <button className="chip" disabled={loading} onClick={() => runClinsim()}
              title="Clinical trial simulation: virtual population across a dose grid → probability of target attainment + dose recommendation">
              Trial simulation (PTA)</button>
          </div>
        )}

        {exportKinds.length > 0 && (
          <div className="quick-actions">
            <span className="quick-actions-label">Export:</span>
            <button className="chip" disabled={loading} onClick={downloadFullReport}>Full report (DOCX)</button>
            {exportKinds.map(k => (
              <button key={k} className="chip" disabled={loading} onClick={() => exportCsv(k)}>
                {EXPORT_LABEL[k] ?? k} CSV
              </button>
            ))}
            {state?.nca_parameters?.length ? (
              <button className="chip" disabled={loading} onClick={exportCdisc}>CDISC ADaM (zip)</button>
            ) : null}
            {state?.nlme_results?.status === 'ok' ? (
              <>
                <button className="chip" disabled={loading} onClick={() => exportControl('nonmem')}
                  title="NONMEM control stream (.ctl) seeded from the population fit">NONMEM (.ctl)</button>
                <button className="chip" disabled={loading} onClick={() => exportControl('mrgsolve')}
                  title="mrgsolve model (.cpp) seeded from the population fit">mrgsolve (.cpp)</button>
              </>
            ) : null}
          </div>
        )}

        {(state?.nca_parameters?.length || state?.nlme_results?.status === 'ok') && (
          <div className="quick-actions">
            <span className="quick-actions-label">Review &amp; skills:</span>
            <button className="chip" disabled={loading} onClick={runReview}
              title="Adversarial reviewer: independently recompute and challenge every result; loop to a checkable goal">
              Adversarial review
            </button>
            <button className="chip" disabled={loading} onClick={captureSkill}
              title="Capture this session's analysis sequence as a reusable, replayable skill">
              Capture as skill
            </button>
            <button className="chip" disabled={loading}
              onClick={() => { setShowSkills(s => !s); if (!showSkills) refreshSkills(); }}>
              {showSkills ? 'Hide skills' : `Skills${skills.length ? ` (${skills.length})` : ''}`}
            </button>
          </div>
        )}

        {showSkills && (
          <SkillsPanel skills={skills} loading={loading}
            datasetPath={state?.dataset_path ?? null}
            onRun={runSkill} onDelete={deleteSkill} onMarkdown={n => api.skillMarkdown(n)} />
        )}

        {hasData && (
          <div className="quick-actions">
            <span className="quick-actions-label">Run on this data:</span>
            {QUICK_ACTIONS.map(qa => (
              <button
                key={qa.label}
                className="chip"
                disabled={loading}
                onClick={() => sendChat(qa.msg)}
              >
                {qa.label}
              </button>
            ))}
          </div>
        )}

        {jobNote && (
          <div className="job-note">
            <span className="job-spinner" /> {jobNote}
          </div>
        )}

        <div className="input-bar">
          <textarea
            rows={1}
            placeholder="Ask agents anything — load dataset, compute NCA, run QC…"
            value={input}
            onChange={e => setInput(e.target.value)}
            onKeyDown={onKeyDown}
            disabled={!session || loading}
          />
          <button
            className="btn btn-primary btn-icon"
            disabled={!input.trim() || !session || loading}
            onClick={() => sendChat()}
            title="Send"
          >
            {loading
              ? <Loader2 size={14} style={{ animation: 'spin 0.6s linear infinite' }} />
              : <Send size={14} />}
          </button>
        </div>
      </main>
    </>
  );
}
