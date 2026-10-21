import { useEffect, useRef, useState, useCallback, useMemo } from 'react';
import type { DragEvent, ChangeEvent } from 'react';
import {
  FlaskConical, Upload, CheckCircle, XCircle,
  AlertTriangle, ChevronRight, Maximize2,
} from 'lucide-react';
import { api, setToken, getToken } from './api';
import { FlexplotPanel } from './flexplot';
import type { Decision, GateSigner, Marker, Outcome, VerifyNote, WorkflowName } from './shell/types';
import { MARKERS, REVIEW_DRAWER_MQ } from './shell/types';
import { hhmm, integrityFailures, latestEntryFor, sealedDecision, stepTimings } from './shell/format';
import { agentLabel } from './shell/agents';
import { Topbar } from './shell/Topbar';
import { Rail } from './shell/Rail';
import { ReviewPanel } from './shell/ReviewPanel';
import { ResultSection } from './shell/ResultSection';
import { DocumentHeader } from './shell/DocumentHeader';
import { DatasetProfile } from './shell/DatasetProfile';
import { ExportMenu } from './shell/ExportMenu';
import { Composer } from './shell/Composer';
import { ActionsMenu } from './shell/ActionsMenu';
import { QcEvidence } from './shell/QcEvidence';
import { PkModelEvidence } from './shell/PkModelEvidence';
import type {
  Session, PharmState, ChatMessage, AuditEntry, AuditIntegrityStatus,
  WorkflowStatus, ContentBlock, PkModelDef, ReviewResults, ReviewFinding, Severity, SkillDef,
  SpaghettiData, NcaPlotData, LzSubject, SimestReplicate, WorkflowResponse, WorkflowExecutedStep,
  PcVpcBin, SpecialPopMetric, SpecialPopStratum, PediatricMetric, PediatricStratum, ProfileParam, WorkflowStartResponse, LlmProvider, LlmConfig,
} from './types';

/** Section title + attribution for every transcript result-card marker.
 *  Typed against the MARKERS tuple, so tsc fails if a marker loses its entry.
 *  `tool` is set ONLY where the backend registers a Tool of that exact name
 *  and the workflow/endpoint seals an audit entry under it (verified in
 *  backend/app/tools/*.py + workflows.py); the section then shows that
 *  entry's hash. Markers without a confirmed tool get agent-only attribution
 *  — never a client-side hash. `anchor` gives the section a DOM id. */
const CARD_META: Record<Marker, { title: string; agent: string; tool?: string; anchor?: string }> = {
  __SPAGHETTI__:     { title: 'Concentration-time',                 agent: 'data_manager', tool: 'generate_spaghetti_plot' },
  __NCA_TABLE__:     { title: 'Non-compartmental analysis',         agent: 'nca',          tool: 'compute_nca' },
  __NCA_LZ__:        { title: 'Terminal slope (λz)',                agent: 'nca',          tool: 'compute_nca', anchor: 'lz' },
  __QC_CARD__:       { title: 'QC review',                          agent: 'qc',           tool: 'run_qc' },
  __BE__:            { title: 'Bioequivalence',                     agent: 'be' },
  __DP__:            { title: 'Dose proportionality',               agent: 'dose_prop' },
  __CLINPHARM__:     { title: 'Clinical pharmacology calculator',   agent: 'clinpharm' },
  __STATS__:         { title: 'Statistical advice',                 agent: 'statistician' },
  __COMPARTMENTAL__: { title: 'Compartmental fit',                  agent: 'compartmental' },
  __POPPK__:         { title: 'Population PK summary',              agent: 'poppk' },
  __PENDING_TOOL__:  { title: 'Proposal · needs your confirmation', agent: 'simulator' },
  __PKMODEL__:       { title: 'Structural model fit',               agent: 'modeler',      tool: 'fit_pk_model', anchor: 'pk-model' },
  __VPC__:           { title: 'VPC / goodness-of-fit',              agent: 'modeler',      tool: 'run_vpc' },
  __NLME__:          { title: 'Population (NLME) fit',              agent: 'modeler',      tool: 'run_nlme' },
  __PRIORCHECK__:    { title: 'Prior check (Bayesian borrowing)',   agent: 'modeler' },
  __SCM__:           { title: 'Covariate model (SCM)',              agent: 'modeler',      tool: 'run_scm' },
  __ENGINES__:       { title: 'Cross-engine comparison',            agent: 'modeler',      tool: 'run_engine_comparison' },
  __FORECAST__:      { title: 'MAP / TDM forecast',                 agent: 'modeler' },
  __DIAG__:          { title: 'Residual diagnostics',               agent: 'modeler',      tool: 'run_diagnostics' },
  __FOREST__:        { title: 'Covariate forest',                   agent: 'modeler',      tool: 'run_covariate_forest' },
  __SIMEST__:        { title: 'Trial-design precision check',       agent: 'simulator',    tool: 'run_simest' },
  __BOOTSTRAP__:     { title: 'Bootstrap uncertainty',              agent: 'simulator',    tool: 'run_bootstrap' },
  __SIR__:           { title: 'SIR uncertainty',                    agent: 'simulator',    tool: 'run_sir' },
  __PROFILE__:       { title: 'Likelihood profiling',               agent: 'simulator',    tool: 'run_profile' },
  __SWEEP__:         { title: 'Dose sweep',                         agent: 'simulator' },
  __CLINSIM__:       { title: 'Clinical trial simulation',          agent: 'simulator' },
  __EXPFOREST__:     { title: 'Exposure covariate forest',          agent: 'simulator' },
  __SPECIALPOP__:    { title: 'Special-population simulation',      agent: 'simulator' },
  __INDIVEXP__:      { title: 'Individual exposures',               agent: 'simulator' },
  __PEDIATRIC__:     { title: 'Pediatric dose-finding',             agent: 'simulator' },
  __SIM__:           { title: 'Forward simulation',                 agent: 'simulator' },
  __REVIEW__:        { title: 'Adversarial review',                 agent: 'reviewer',     tool: 'adversarial_review' },
  __REPORT__:        { title: 'Report',                             agent: 'report' },
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

/** While a workflow leg runs as a job, re-read the audit trail every this
 *  many poll ticks (1.5 s each) after the first, so the seals the leg adds
 *  reach the rail and the review panel before the leg ends. */
const AUDIT_REFRESH_EVERY_TICKS = 20;

/** Result card each workflow step renders once it has run, keyed by tool, with
 *  the PharmState slot the card reads. Mirrors what the sidebar chips push, so
 *  a leg that ran as a background job (the NLME leg) shows the same cards as
 *  one that ran inline. Steps missing here (and steps whose card is pushed
 *  from state below) get no message: a step with a section is not narrated,
 *  and the rest go to the document's folded run log. */
const STEP_CARD: Record<string, { marker: Marker; key: keyof PharmState }> = {
  run_nlme:             { marker: '__NLME__',   key: 'nlme_results' },
  run_scm:              { marker: '__SCM__',    key: 'scm_results' },
  run_diagnostics:      { marker: '__DIAG__',   key: 'diagnostics_results' },
  run_covariate_forest: { marker: '__FOREST__', key: 'forest_results' },
  run_vpc:              { marker: '__VPC__',    key: 'vpc_results' },
};

/** Human-review banner subtitle: what the gated step just finished and what
 *  approval runs next. Phrased per step key, falling back to the step labels
 *  so a new gate never shows another gate's sentence. */
const GATE_DONE: Record<string, string> = {
  qc_review: 'QC complete',
  fit_pk_model: 'Structural comparison complete',
  adversarial_review: 'Adversarial review complete',
};
const GATE_NEXT: Record<string, string> = {
  generate_report: 'generate the DOCX report',
  run_nlme: 'run the population fit',
};
/** What approval runs, as a chain — named once here, not restated in the title. */
const GATE_CHAIN: Record<string, string> = {
  generate_report: 'DOCX report',
  run_nlme: 'NLME → SCM → diagnostics → forest → VPC',
};
/** Reason placeholder per gated step, so each gate shows an example of its own. */
const GATE_PLACEHOLDER: Record<string, string> = {
  qc_review: 'e.g. Subject 1 extrapolation accepted; sparse terminal sampling',
  fit_pk_model: 'e.g. 1-cmt oral accepted; lowest AIC, all converged',
  adversarial_review: 'e.g. Reviewer concerns addressed; fit accepted',
};

function gateSubtitle(workflow: WorkflowName, gateIndex: number): string {
  const steps = WORKFLOW_UI[workflow].steps;
  const gate = steps[gateIndex];
  if (!gate) return 'Approve to continue the workflow.';
  const done = GATE_DONE[gate.key] ?? `${gate.label} complete`;
  const next = steps[gateIndex + 1];
  return next ? `${done}. Next: ${GATE_CHAIN[next.key] ?? next.label}.` : `${done}.`;
}

/** A step summary that opens with its own label ("Adversarial review: GOAL
 *  MET…") is shown without it: the run log entry is already headed by the
 *  label, so repeating it would read as a doubled prefix. */
function stripLabelPrefix(label: string, summary: string): string {
  const prefix = `${label}:`;
  if (!summary.toLowerCase().startsWith(prefix.toLowerCase())) return summary;
  return summary.slice(prefix.length).trimStart();
}

function fmt(v: number | undefined, d = 2) {
  if (v == null || isNaN(v)) return '–';
  return v.toFixed(d);
}

// `snap` freezes the PharmState slice a result card reads, so re-running an
// analysis later does not retroactively rewrite earlier cards in the transcript.
// `seal` is the audit entry hash bound to that snapshot by `bindSeals` (once,
// from the audit fetched after the card's own response); `seq` is the push
// order pushMsg stamps so a fetch sent earlier can never bind a later card.
type DisplayMsg = {
  role: string; content: string; agent?: string; tool?: string; id: string;
  snap?: PharmState | null; seal?: string; seq?: number;
};
const newMsgId = () => `${Date.now()}-${Math.random()}`;

/** Bind every still-unsealed snapshot card pushed no later than `seqAtFetch`
 *  to the latest audit entry sealed under its tool in `audit` — the entry of
 *  the run that produced the card's snapshot, since that audit was fetched
 *  after the card's response landed. A card is bound once and never rebound,
 *  so a later re-run (new card, new snapshot) leaves the older card's seal on
 *  the older numbers. Returns `msgs` itself when nothing binds. */
function bindSeals(msgs: DisplayMsg[], audit: AuditEntry[], seqAtFetch: number): DisplayMsg[] {
  let changed = false;
  const out = msgs.map(m => {
    if (!m.snap || m.seal !== undefined || (m.seq ?? Infinity) > seqAtFetch) return m;
    const marker = (MARKERS as readonly string[]).includes(m.content) ? m.content as Marker : null;
    const hash = marker ? latestEntryFor(audit, CARD_META[marker].tool)?.entry_hash : undefined;
    if (!hash) return m;
    changed = true;
    return { ...m, seal: hash };
  });
  return changed ? out : msgs;
}

function MessageBubble({ msg, agent }: { msg: DisplayMsg; agent?: string }) {
  if (msg.role === 'user') {
    return <div className="msg user">You · {msg.content}</div>;
  }
  return (
    <div className="msg agent">
      {agent && <span className="msg-agent">{agentLabel(agent)}</span>}
      <div style={{ whiteSpace: 'pre-wrap' }}>{msg.content}</div>
      {msg.tool && (
        <div className="tool-chip">
          <ChevronRight size={10} /> {msg.tool}
        </div>
      )}
    </div>
  );
}

function QcCard({ state, onReviewLz, onReviewInPanel }: { state: PharmState; onReviewLz?: () => void; onReviewInPanel?: () => void }) {
  const v = state.qc_verdict ?? '';
  const cls = v === 'PASS' ? 'pass' : v.includes('CONDITIONAL') ? 'conditional' : 'fail';
  // One verdict line; the checklist itself lives in the review panel.
  return (
    <div className={`qc-card ${cls}`}>
      <QcEvidence verdict={v} checklist={state.qc_checklist} issues={state.qc_issues} subjects={state.nca_parameters}
        onReviewLz={onReviewLz} onReviewInPanel={onReviewInPanel} compact />
    </div>
  );
}

const SEVERITY_COLOR: Record<Severity, string> = {
  CRITICAL: '#B23A2E', HIGH: '#9A5B12', MEDIUM: '#1F66A6', LOW: '#536A83',
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
        <span style={{ color: 'var(--text-dim)', fontWeight: 400 }}> — {r.goal}</span>
      </div>
      <div className="review-counts" style={{ display: 'flex', gap: 10, margin: '6px 0 10px' }}>
        {(['CRITICAL', 'HIGH', 'MEDIUM', 'LOW'] as Severity[]).map(s => (
          <span key={s} style={{ fontSize: 12, color: c[s] ? SEVERITY_COLOR[s] : 'var(--text-dim)' }}>
            {c[s]} {s.toLowerCase()}
          </span>
        ))}
      </div>
      {r.findings.length === 0 ? (
        <div style={{ color: 'var(--text-dim)', fontSize: 13 }}>
          No findings — the reviewer could not refute any reported value.
        </div>
      ) : (
        <ul style={{ listStyle: 'none', padding: 0, margin: 0, display: 'flex', flexDirection: 'column', gap: 8 }}>
          {r.findings.map((f: ReviewFinding) => (
            <li key={f.id}>
              <div style={{ fontSize: 12 }}>
                <span className={`sev-pill ${f.severity}`}>{f.severity}</span>
                <span style={{ color: 'var(--text)' }}> · {f.target}</span>
              </div>
              <div style={{ fontSize: 13, marginTop: 2 }}><strong>Claim:</strong> {f.claim}</div>
              <div style={{ fontSize: 13 }}><strong>Evidence:</strong> {f.evidence}</div>
              <div style={{ fontSize: 12, color: 'var(--text-dim)', marginTop: 2 }}>→ {f.suggested_action}</div>
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
        <div style={{ color: 'var(--text-dim)', fontSize: 13 }}>
          No skills captured yet. Run an analysis, then “Capture as skill”.
        </div>
      ) : (
        <ul style={{ listStyle: 'none', padding: 0, margin: 0, display: 'flex', flexDirection: 'column', gap: 6 }}>
          {skills.map(s => (
            <li key={s.name} style={{ display: 'flex', alignItems: 'center', gap: 8, flexWrap: 'wrap' }}>
              <span style={{ fontWeight: 600 }}>{s.name}</span>
              <span style={{ fontSize: 12, color: 'var(--text-dim)' }}>v{s.version}</span>
              <span style={{ fontSize: 12, fontFamily: 'var(--mono)', color: 'var(--text-dim)' }}>{s.steps.map(st => st.tool).join(' → ')}</span>
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

/** %AUC extrapolation above this is flagged in the NCA table, the λz panels
 *  and the concentration-time chart (same limit the QC agent applies). */
const EXTRAP_LIMIT = 20;
/** Subject id → %AUC extrapolated, for every subject over EXTRAP_LIMIT.
 *  Empty until NCA has run. */
function flaggedExtrap(st: PharmState | null | undefined): Record<string, number> {
  const out: Record<string, number> = {};
  for (const r of st?.nca_parameters ?? []) {
    if ((r.pct_AUC_extrap ?? 0) > EXTRAP_LIMIT) out[String(r.subject)] = r.pct_AUC_extrap as number;
  }
  return out;
}
/** Natural order for subject ids, so "2" sorts before "10". Returns a new array. */
function bySubjectId<T>(items: readonly T[], id: (t: T) => unknown): T[] {
  return [...items].sort((a, b) =>
    String(id(a)).localeCompare(String(id(b)), undefined, { numeric: true, sensitivity: 'base' }));
}
/** Rows shown before "Show all N subjects". */
const NCA_ROWS_COLLAPSED = 4;
const NCA_ROWS_THRESHOLD = 6;

function ExtrapBadge() {
  return (
    <span className="badge-warn" title="Over the 20% limit" aria-label="over the 20 percent limit">!</span>
  );
}

function NcaSubjectTable({ state, limit }: { state: PharmState; limit?: number }) {
  const raw = state.nca_parameters;
  if (!raw || raw.length === 0) return null;
  const rows = bySubjectId(raw, r => r.subject);
  const shown = limit != null ? rows.slice(0, limit) : rows;
  const ss = state.nca_summary?.steady_state === true;
  const tau = rows.find(r => r.tau != null)?.tau;
  const u = ncaUnits(state);

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
              <th>ID</th><th className="num">Dose{unitSuffix(u.dose)}</th><th className="num">Cmax,ss{unitSuffix(u.conc)}</th><th className="num">Cmin{unitSuffix(u.conc)}</th>
              <th className="num">AUC<sub>τ</sub>{unitSuffix(u.auc)}</th><th className="num">Cavg{unitSuffix(u.conc)}</th>
              <th className="num">CL/F{unitSuffix(u.clf)}</th><th className="num">t½{unitSuffix(u.time)}</th><th className="num">Fluct%</th><th className="num">R<sub>ac</sub></th>
            </tr>
          </thead>
          <tbody>
            {shown.map(r => (
              <tr key={String(r.subject)}>
                <td>{r.subject}</td>
                <td className="num">{fmt(r.dose, 0)}</td>
                <td className="num">{fmt(r.Cmax)}</td>
                <td className="num">{fmt(r.Cmin ?? undefined)}</td>
                <td className="num">{fmt(r.AUC_tau ?? r.AUC_last, 1)}</td>
                <td className="num">{fmt(r.Cavg ?? undefined, 1)}</td>
                <td className="num">{fmt(r.CL_F, 2)}</td>
                <td className="num">{fmt(r.t_half, 1)}</td>
                <td className="num">{fmt(r.fluctuation_pct ?? undefined, 0)}</td>
                <td className="num">{fmt(r.accumulation_ratio ?? undefined, 2)}</td>
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
            <th>ID</th><th className="num">Dose{unitSuffix(u.dose)}</th><th className="num">Cmax{unitSuffix(u.conc)}</th><th className="num">Tmax{unitSuffix(u.time)}</th>
            <th className="num">AUC<sub>last</sub>{unitSuffix(u.auc)}</th><th className="num">AUC<sub>inf</sub>{unitSuffix(u.auc)}</th>
            <th className="num">t½{unitSuffix(u.time)}</th><th className="num">CL/F{unitSuffix(u.clf)}</th><th className="num">Vz/F{unitSuffix(u.vz)}</th><th className="num">% extrap.</th>
          </tr>
        </thead>
        <tbody>
          {shown.map(r => {
            const hi = (r.pct_AUC_extrap ?? 0) > EXTRAP_LIMIT;
            return (
              <tr key={String(r.subject)}>
                <td>{r.subject}</td>
                <td className="num">{fmt(r.dose, 0)}</td>
                <td className="num">{fmt(r.Cmax)}</td>
                <td className="num">{fmt(r.Tmax)}</td>
                <td className="num">{fmt(r.AUC_last, 1)}</td>
                <td className="num">{fmt(r.AUC_inf ?? undefined, 1)}</td>
                <td className="num">{fmt(r.t_half, 1)}</td>
                <td className="num">{fmt(r.CL_F, 2)}</td>
                <td className="num">{fmt(r.Vz_F, 1)}</td>
                <td className={`num ${hi ? 'extrap-hi' : ''}`}>
                  {fmt(r.pct_AUC_extrap ?? undefined, 1)}%{hi && <> <ExtrapBadge /></>}
                </td>
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

const NCA_PARAM_LABEL: Record<string, string> = {
  Cmax: 'Cmax', Tmax: 'Tmax', Cmin: 'Cmin', Ctrough: 'Ctrough', Cavg: 'Cavg', Clast: 'Clast', Tlast: 'Tlast',
  AUC_last: 'AUClast', AUC_inf: 'AUCinf', AUC_tau: 'AUCτ', pct_AUC_extrap: '%AUC extrap',
  lambda_z: 'λz', t_half: 't½', MRT: 'MRT', CL_F: 'CL/F', Vz_F: 'Vz/F', Vss: 'Vss',
  accumulation_ratio: 'Rac', fluctuation_pct: 'Fluctuation %', swing_pct: 'Swing %',
};

/** Standard NCA summary statistics per parameter (N, mean, SD, CV%, median,
 *  min, max, geometric mean, geo-CV%), overall or per dose group. */
function DescriptiveStatsTable({ state }: { state: PharmState }) {
  const groups = state.nca_summary?.descriptive ?? [];
  const [gi, setGi] = useState(0);
  if (!groups.length) return null;
  const g = groups[Math.min(gi, groups.length - 1)];
  const d = (v: number | null, digits = 3) => (v == null ? '–' : fmt(v, digits));
  const pct = (v: number | null) => (v == null ? '–' : `${fmt(v, 1)}%`);
  return (
    <div style={{ marginTop: 12 }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 4 }}>
        <span style={{ fontSize: 12, color: 'var(--text-dim)' }}>Summary statistics</span>
        {groups.length > 1 && (
          <select value={gi} onChange={e => setGi(Number(e.target.value))} aria-label="Summary statistics group"
            style={{ fontSize: 12, padding: '2px 6px', borderRadius: 6, border: '1px solid var(--border-strong)', background: 'var(--bg-card)', color: 'var(--text-h)' }}>
            {groups.map((x, i) => <option key={String(x.group)} value={i}>{x.label} (n={x.n})</option>)}
          </select>
        )}
        {groups.length === 1 && <span style={{ fontSize: 11, color: 'var(--text-dim)' }}>· {g.label} (n={g.n})</span>}
      </div>
      <div style={{ overflowX: 'auto' }}>
        <table className="nca-table">
          <thead>
            <tr><th>Parameter</th><th>N</th><th>Mean</th><th>SD</th><th>CV%</th><th>Median</th><th>Min</th><th>Max</th><th>Geo. mean</th><th>Geo. CV%</th></tr>
          </thead>
          <tbody>
            {g.parameters.map(p => (
              <tr key={p.parameter}>
                <td>{NCA_PARAM_LABEL[p.parameter] ?? p.parameter}</td>
                <td>{p.n}</td>
                <td>{d(p.mean)}</td><td>{d(p.sd)}</td><td>{pct(p.cv_pct)}</td>
                <td>{d(p.median)}</td><td>{d(p.min)}</td><td>{d(p.max)}</td>
                <td>{d(p.geomean)}</td><td>{pct(p.geocv_pct)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div style={{ fontSize: 11, color: 'var(--text-dim)', marginTop: 4 }}>
        SD uses n−1; CV% = SD/mean; geometric CV% = √(exp(s²<sub>log</sub>) − 1); N counts subjects with a valid value.
      </div>
    </div>
  );
}

/** 'h' after a time-based value only when the dataset's time axis is
 *  labelled in hours — units are never invented. */
function timeInHours(st: PharmState | null | undefined): boolean {
  return /hour|\(h\)|\bh\b/i.test(st?.spaghetti_data?.x_label ?? '');
}

/** Units as stated by the dataset's own column labels (e.g. "Dose (mg)",
 *  "Conc(mg/L)", "Time(h)") — never invented. Derived NCA units (AUC, CL/F,
 *  Vz/F) are stated only when they follow unambiguously from those labels. */
interface NcaUnits {
  time: string | null; dose: string | null; conc: string | null;
  auc: string | null; clf: string | null; vz: string | null;
}

function roleColumnUnit(st: PharmState | null | undefined, role: string): string | null {
  const roles = st?.dataset_metadata?.detected_roles;
  if (!roles || typeof roles !== 'object') return null;
  for (const [col, r] of Object.entries(roles as Record<string, unknown>)) {
    const u = r === role ? labelUnit(col) : null;
    if (u) return u;
  }
  return null;
}

function ncaUnits(st: PharmState | null | undefined): NcaUnits {
  const time = timeInHours(st) ? 'h'
    : labelUnit(st?.spaghetti_data?.x_label ?? '') ?? roleColumnUnit(st, 'TIME');
  const conc = labelUnit(st?.spaghetti_data?.y_label ?? '') ?? roleColumnUnit(st, 'DV');
  const dose = roleColumnUnit(st, 'AMT');
  const slash = conc ? conc.indexOf('/') : -1;
  const [cNum, cDen] = conc && slash > 0 ? [conc.slice(0, slash), conc.slice(slash + 1)] : [conc, null];
  const auc = conc && time ? (cDen ? `${cNum}·${time}/${cDen}` : `${conc}·${time}`) : null;
  const volume = dose && cDen && cNum === dose ? cDen : null;
  return { time, dose, conc, auc, clf: volume && time ? `${volume}/${time}` : null, vz: volume };
}

/** " (unit)" for a column header, or nothing when the unit is unknown. */
function unitSuffix(u: string | null): string {
  return u ? ` (${u})` : '';
}

/** 3 significant figures, no exponent notation (267.84 → "268"). */
function sig3(v: number): string {
  return Number(v.toPrecision(3)).toString();
}

/** One-line geometric-mean summary under the NCA table, built ONLY from the
 *  fields present in nca_summary.descriptive (group 'all' or the first
 *  group). Units are never invented: each comes from ncaUnits (the dataset's
 *  own column labels). Returns null when nothing can be stated. */
function ncaFooterParts(state: PharmState): React.ReactNode[] {
  const groups = state.nca_summary?.descriptive ?? [];
  const g = groups.find(x => x.group === 'all') ?? groups[0];
  if (!g) return [];
  const param = (name: string) => g.parameters.find(p => p.parameter === name);
  const u = ncaUnits(state);
  const parts: React.ReactNode[] = [];
  const cl = param('CL_F');
  if (cl?.geomean != null) {
    parts.push(<span key="cl">Geometric mean CL/F <b>{fmt(cl.geomean, 3)}{u.clf ? ` ${u.clf}` : ''}</b>{cl.geocv_pct != null && <> (gCV {fmt(cl.geocv_pct, 1)} %)</>}</span>);
  }
  const th = param('t_half');
  if (th?.median != null) {
    parts.push(<span key="th">t½ <b>{fmt(th.median, 1)}{u.time ? ` ${u.time}` : ''}</b></span>);
  }
  const auc = param('AUC_inf');
  if (auc?.geomean != null) {
    parts.push(<span key="auc">AUCinf <b>{fmt(auc.geomean, 1)}{u.auc ? ` ${u.auc}` : ''}</b>{auc.geocv_pct != null && <> (gCV {fmt(auc.geocv_pct, 1)} %)</>}</span>);
  }
  return parts;
}

function NcaTable({ state }: { state: PharmState }) {
  const n = state.nca_parameters?.length ?? 0;
  const collapsible = n > NCA_ROWS_THRESHOLD;
  const [showAll, setShowAll] = useState(false);
  const [showStats, setShowStats] = useState(false);
  const hasStats = !!state.nca_summary?.descriptive?.length || !!state.nca_summary?.by_dose?.length;
  const footer = ncaFooterParts(state);
  return (
    <div>
      <NcaSubjectTable state={state} limit={collapsible && !showAll ? NCA_ROWS_COLLAPSED : undefined} />
      {(footer.length > 0 || collapsible || hasStats) && (
        <div className="nca-foot">
          <span className="nca-foot-stats">
            {footer.map((f, i) => <span key={i}>{i > 0 && ' · '}{f}</span>)}
          </span>
          <span className="nca-foot-links">
            {collapsible && (
              <button type="button" className="link-btn" onClick={() => setShowAll(v => !v)}>
                {showAll ? 'Show fewer' : `Show all ${n} subjects`}
              </button>
            )}
            {hasStats && (
              <button type="button" className="link-btn" aria-expanded={showStats} onClick={() => setShowStats(v => !v)}>
                Summary statistics
              </button>
            )}
          </span>
        </div>
      )}
      {showStats && (
        <>
          <DescriptiveStatsTable state={state} />
          <DoseSummaryTable state={state} />
        </>
      )}
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

const ci = (lo: number | null | undefined, hi: number | null | undefined, d = 3) =>
  lo == null || hi == null ? '–' : `[${fmt(lo, d)}, ${fmt(hi, d)}]`;

function UncertaintyNotes({ notes }: { notes?: string[] }) {
  if (!notes?.length) return null;
  return (
    <ul style={{ fontSize: 11, color: 'var(--text-dim)', margin: '8px 0 0', paddingLeft: 16 }}>
      {notes.map((n, i) => <li key={i}>{n}</li>)}
    </ul>
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

const SPAG_PALETTE = ['#1F66A6','#1D7A5A','#9A5B12','#4A6FA5','#B23A2E','#3B86C9','#16604A','#C77F2A','#5E7388','#2A8F8F'];

/** The API wraps failures as {"error":{"message"}} (or {"detail"}); show the message, not the JSON. */
function errorText(e: unknown): string {
  const raw = (e as Error).message ?? String(e);
  const start = raw.indexOf('{');
  if (start >= 0) {
    try {
      const j = JSON.parse(raw.slice(start)) as { error?: { message?: string }; detail?: string; message?: string };
      const m = j.error?.message ?? j.detail ?? j.message;
      if (m) return m;
    } catch { /* not JSON */ }
  }
  return raw;
}

const PROVIDER_LABEL: Record<LlmProvider, string> = {
  mock: 'Mock (keyless, deterministic)', local: 'Local — Ollama (free)',
  openai: 'ChatGPT — OpenAI API key', anthropic: 'Claude — Anthropic API key',
};

/** Header popover: pick the model that routes requests and picks tools.
 *  Keys are sent to the local backend and held in its memory only. */
function LlmSettings({ onApplied, onClose }: { onApplied: (label: string) => void; onClose: () => void }) {
  const [cfg, setCfg] = useState<LlmConfig | null>(null);
  const [provider, setProvider] = useState<LlmProvider>('mock');
  const [model, setModel] = useState('');
  const [apiKey, setApiKey] = useState('');
  const [baseUrl, setBaseUrl] = useState('');
  const [busy, setBusy] = useState<'test' | 'apply' | null>(null);
  const [note, setNote] = useState<{ ok: boolean; text: string } | null>(null);

  useEffect(() => {
    api.getLlm().then(c => {
      setCfg(c); setProvider(c.current.provider); setModel(c.current.model);
      setBaseUrl(c.current.provider === 'local' && c.current.base_url ? c.current.base_url : '');
    }).catch(e => setNote({ ok: false, text: (e as Error).message }));
  }, []);

  const pick = (p: LlmProvider) => {
    setProvider(p); setNote(null); setApiKey('');
    setModel(cfg && cfg.current.provider === p ? cfg.current.model : (cfg?.defaults[p] ?? ''));
  };
  const needsKey = provider === 'openai' || provider === 'anthropic';
  const keyHeld = !!cfg && cfg.current.provider === provider && cfg.current.has_key;

  const submit = async (testOnly: boolean) => {
    setBusy(testOnly ? 'test' : 'apply'); setNote(null);
    try {
      const r = await api.setLlm({ provider, model, ...(apiKey ? { api_key: apiKey } : {}),
        ...(baseUrl ? { base_url: baseUrl } : {}), test_only: testOnly });
      setNote({ ok: true, text: testOnly ? `Test OK — ${r.detail}` : `Switched — ${r.detail}` });
      if (!testOnly) { setApiKey(''); const c = await api.getLlm(); setCfg(c); onApplied(c.label); }
    } catch (e) {
      setNote({ ok: false, text: errorText(e) });
    } finally { setBusy(null); }
  };

  const field = { width: '100%', fontSize: 12, padding: '5px 8px', borderRadius: 6,
    border: '1px solid var(--border)', background: 'var(--bg)', color: 'var(--text)' } as const;
  return (
    <div style={{ width: 360, padding: 2, background: 'var(--bg-card)', textAlign: 'left' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8 }}>
        <strong style={{ fontSize: 13 }}>Language model</strong>
        <button className="btn btn-ghost" style={{ padding: '2px 8px', fontSize: 11 }} onClick={onClose}>close</button>
      </div>
      <div style={{ fontSize: 11, color: 'var(--text-dim)', marginBottom: 8 }}>
        Only routing and tool choice use the model; every number comes from deterministic code.
        API keys stay in the local backend&apos;s memory (never on disk, never shown again).
      </div>
      {(cfg?.providers ?? (['mock', 'local', 'openai', 'anthropic'] as LlmProvider[])).map(p => (
        <label key={p} style={{ display: 'flex', gap: 8, alignItems: 'center', fontSize: 12, padding: '3px 0' }}>
          <input type="radio" name="llm-provider" checked={provider === p} onChange={() => pick(p)} />
          {PROVIDER_LABEL[p]}
          {cfg?.current.provider === p && <span style={{ color: 'var(--green)', fontSize: 11 }}>· active</span>}
        </label>
      ))}
      {provider !== 'mock' && (
        <div style={{ marginTop: 8 }}>
          <div style={{ fontSize: 11, color: 'var(--text-dim)', marginBottom: 2 }}>Model</div>
          <input style={field} value={model} onChange={e => setModel(e.target.value)} list="llm-local-models"
            placeholder={cfg?.defaults[provider]} />
          {provider === 'local' && (
            <datalist id="llm-local-models">{(cfg?.local_models ?? []).map(m => <option key={m} value={m} />)}</datalist>
          )}
          {provider === 'local' && !(cfg?.local_models ?? []).length && (
            <div style={{ fontSize: 11, color: 'var(--yellow)', marginTop: 3 }}>
              No Ollama models found — run <code>ollama pull gemma4:31b</code> (Ollama must be running).
            </div>
          )}
        </div>
      )}
      {needsKey && (
        <div style={{ marginTop: 8 }}>
          <div style={{ fontSize: 11, color: 'var(--text-dim)', marginBottom: 2 }}>
            API key {keyHeld ? '(one is held — leave blank to keep it)' : ''}
          </div>
          <input style={field} type="password" value={apiKey} onChange={e => setApiKey(e.target.value)}
            placeholder={provider === 'openai' ? 'sk-…' : 'sk-ant-…'} autoComplete="off" />
        </div>
      )}
      {(provider === 'local' || provider === 'openai') && (
        <div style={{ marginTop: 8 }}>
          <div style={{ fontSize: 11, color: 'var(--text-dim)', marginBottom: 2 }}>Server URL (optional)</div>
          <input style={field} value={baseUrl} onChange={e => setBaseUrl(e.target.value)}
            placeholder={provider === 'local' ? 'http://127.0.0.1:11434/v1 (Ollama) · LM Studio: http://127.0.0.1:1234/v1' : 'https://api.openai.com/v1'} />
        </div>
      )}
      <div style={{ display: 'flex', gap: 6, marginTop: 10 }}>
        <button className="btn btn-ghost" disabled={busy !== null} onClick={() => submit(true)}>
          {busy === 'test' ? 'Testing…' : 'Test'}
        </button>
        <button className="btn btn-green" disabled={busy !== null} onClick={() => submit(false)}>
          {busy === 'apply' ? 'Switching…' : 'Use this model'}
        </button>
      </div>
      {note && (
        <div style={{ fontSize: 11, marginTop: 8, color: note.ok ? 'var(--green)' : 'var(--red)', wordBreak: 'break-word' }}>
          {note.text}
        </div>
      )}
    </div>
  );
}


/** Hover a chart → ⤢ opens it full-screen with wheel zoom and drag-to-pan.
 *  The same JSX is rendered again inside the modal, so nothing is rasterised. */
function Zoomable({ title, style, children }: { title: string; style?: React.CSSProperties; children: React.ReactNode }) {
  const [open, setOpen] = useState(false);
  return (
    <div className="zoomable" style={style}>
      {children}
      <button className="zoom-btn" type="button" aria-label="Open full screen" title={`Open ${title} full screen — wheel to zoom, drag to pan`}
        onClick={() => setOpen(true)}><Maximize2 size={13} aria-hidden="true" /></button>
      {open && <ZoomModal title={title} onClose={() => setOpen(false)}>{children}</ZoomModal>}
    </div>
  );
}

function ZoomModal({ title, onClose, children }: { title: string; onClose: () => void; children: React.ReactNode }) {
  const [z, setZ] = useState(1);
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const [dragging, setDragging] = useState(false);
  const drag = useRef<{ x: number; y: number; px: number; py: number } | null>(null);
  const bodyRef = useRef<HTMLDivElement>(null);
  const [width, setWidth] = useState(900);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose();
      if (e.key === '+' || e.key === '=') setZ(v => Math.min(8, v * 1.2));
      if (e.key === '-') setZ(v => Math.max(0.2, v / 1.2));
      if (e.key === '0') { setZ(1); setPan({ x: 0, y: 0 }); }
    };
    window.addEventListener('keydown', onKey);
    const el = bodyRef.current;
    // fit the whole chart: width-limited OR height-limited via the SVG's viewBox aspect
    const fit = () => {
      if (!el) return;
      const vb = el.querySelector('svg')?.getAttribute('viewBox')?.split(/\s+/).map(Number);
      const aspect = vb && vb.length === 4 && vb[3] > 0 ? vb[2] / vb[3] : 2;
      setWidth(Math.max(320, Math.min(el.clientWidth * 0.92, el.clientHeight * 0.92 * aspect)));
    };
    fit();
    window.addEventListener('resize', fit);
    // non-passive so the page does not scroll behind the modal
    const onWheel = (e: WheelEvent) => { e.preventDefault(); setZ(v => Math.min(8, Math.max(0.2, v * (e.deltaY < 0 ? 1.1 : 1 / 1.1)))); };
    el?.addEventListener('wheel', onWheel, { passive: false });
    return () => { window.removeEventListener('keydown', onKey); window.removeEventListener('resize', fit); el?.removeEventListener('wheel', onWheel); };
  }, [onClose]);

  const down = (e: React.MouseEvent) => { drag.current = { x: e.clientX, y: e.clientY, px: pan.x, py: pan.y }; setDragging(true); };
  const move = (e: React.MouseEvent) => {
    if (!drag.current) return;
    setPan({ x: drag.current.px + (e.clientX - drag.current.x), y: drag.current.py + (e.clientY - drag.current.y) });
  };
  const up = () => { drag.current = null; setDragging(false); };

  return (
    <div className="zoom-overlay" onClick={onClose} role="dialog" aria-label={`${title} — enlarged`}>
      <div className="zoom-modal" onClick={e => e.stopPropagation()}>
        <div className="zoom-toolbar">
          <span className="zoom-title">{title}</span>
          <span style={{ color: 'var(--text-dim)' }}>{Math.round(z * 100)}%</span>
          <button className="btn btn-ghost" onClick={() => setZ(v => Math.max(0.2, v / 1.2))} title="Zoom out (−)">−</button>
          <button className="btn btn-ghost" onClick={() => setZ(v => Math.min(8, v * 1.2))} title="Zoom in (+)">+</button>
          <button className="btn btn-ghost" onClick={() => { setZ(1); setPan({ x: 0, y: 0 }); }} title="Reset (0)">reset</button>
          <button className="btn btn-ghost" onClick={onClose} title="Close (Esc)">close</button>
        </div>
        <div ref={bodyRef} className={`zoom-body${dragging ? ' dragging' : ''}`}
          onMouseDown={down} onMouseMove={move} onMouseUp={up} onMouseLeave={up}>
          <div className="zoom-canvas" style={{ width, transform: `translate(-50%, -50%) translate(${pan.x}px, ${pan.y}px) scale(${z})` }}>
            {children}
          </div>
        </div>
        <div style={{ fontSize: 11, color: 'var(--text-dim)', padding: '4px 12px', borderTop: '1px solid var(--border)' }}>
          wheel / + − to zoom · drag to pan · 0 resets · Esc closes
        </div>
      </div>
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

/** /api/health "llm" -> badge text: "mock" | "anthropic:<model>" | "openai:<model>@<url>". */
function describeLlm(label: string): string {
  if (!label || label === 'mock') return 'Keyless mock model';
  const [provider, rest = ''] = label.split(/:(.+)/);
  if (provider === 'anthropic') return `Claude ${rest}`;
  if (provider === 'openai') {
    const [model, url = ''] = rest.split('@');
    if (url.includes('api.openai.com')) return `ChatGPT ${model}`;
    const host = url.includes('127.0.0.1') || url.includes('localhost') ? 'local' : url.replace(/^https?:\/\//, '').split('/')[0];
    return `${model} (${host})`;
  }
  return label;
}

/** "Nice" tick values across [lo, hi] (1/2/5 × 10^k steps), like d3/flexplot. */
function niceTicks(lo: number, hi: number, count = 5): number[] {
  if (!(hi > lo)) return [lo];
  const raw = (hi - lo) / count, mag = Math.pow(10, Math.floor(Math.log10(raw)));
  const norm = raw / mag;
  const step = (norm >= 5 ? 5 : norm >= 2 ? 2 : 1) * mag;
  const out: number[] = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-9; v += step) out.push(+v.toFixed(10));
  return out;
}

/** Axis titles were 9 px in the dim text colour — unreadable once the SVG is
 *  scaled into a chat bubble. Titles use the normal text colour, ticks stay dim. */
const AXIS_TITLE = { fontSize: 11, fontWeight: 600, fill: 'var(--text)' } as const;

/** Two-thumb time window: from / to sliders over the data's x-range plus a
 *  numeric readout and reset. Values are clamped and ordered by the caller. */
function XRangeSlider({ min, max, value, label, onChange, onReset }: {
  min: number; max: number; value: [number, number]; label: string;
  onChange: (v: [number, number]) => void; onReset: () => void;
}) {
  const span = max - min || 1;
  const step = span / 200;
  const isFull = value[0] <= min + 1e-9 && value[1] >= max - 1e-9;
  const num = (v: number) => (Math.abs(v) >= 100 ? v.toFixed(0) : v.toFixed(1));
  const slider = { width: 110, verticalAlign: 'middle', accentColor: 'var(--accent)' } as const;
  return (
    <span style={{ display: 'inline-flex', alignItems: 'center', gap: 6, fontSize: 11, color: 'var(--text-dim)', flexWrap: 'wrap' }}
      title="Restrict the time axis to a window">
      <span>{label}</span>
      <input type="range" min={min} max={max} step={step} value={value[0]} style={slider} aria-label="window start"
        onChange={e => onChange([Math.min(Number(e.target.value), value[1]), value[1]])} />
      <span style={{ fontFamily: 'var(--mono)', color: 'var(--text)' }}>{num(value[0])}–{num(value[1])}</span>
      <input type="range" min={min} max={max} step={step} value={value[1]} style={slider} aria-label="window end"
        onChange={e => onChange([value[0], Math.max(Number(e.target.value), value[0])])} />
      {!isFull && <button type="button" className="link-btn" onClick={onReset}>Reset time window</button>}
    </span>
  );
}

/** Sentence for the chart caption naming the subjects whose terminal phase
 *  is flagged in the NCA table below. */
function flaggedSentence(ids: string[]): string {
  if (ids.length === 0) return '';
  if (ids.length === 1) return ` · subject ${ids[0]} highlighted because its terminal phase is flagged below`;
  return ` · subjects ${ids.join(', ')} highlighted because their terminal phases are flagged below`;
}

const TICK_TEXT = { fontFamily: 'var(--mono)', fill: 'var(--text-dim)' } as const;

/** Unit written inside the raw column label, e.g. 'DV (mg/L)' or 'TIME [h]'. */
function labelUnit(raw: string): string | null {
  const m = /[([]\s*([^)\]]+?)\s*[)\]]/.exec(raw);
  return m && !/^log$/i.test(m[1]) ? m[1] : null;
}

/** Sentence-case axis titles for the concentration–time chart. The unit is
 *  shown only when the raw column label states it — never invented. */
function spaghettiAxisLabels(data: SpaghettiData): { x: string; y: string } {
  const xUnit = labelUnit(data.x_label);
  const hours = /hour|\(h\)|\bh\b/i.test(data.x_label);
  const yUnit = labelUnit(data.y_label);
  return {
    x: `Time${hours ? ' (h)' : xUnit ? ` (${xUnit})` : ''}`,
    y: `Concentration${yUnit ? ` (${yUnit})` : ''}`,
  };
}

function SpaghettiChart({ data, flagged = [] }: { data: SpaghettiData; flagged?: string[] }) {
  const [logY, setLogY] = useState(data.log_scale);
  const [individual, setIndividual] = useState(false);
  // user-chosen time window (null = the data's full range); applies to both views
  const [xr, setXr] = useState<[number, number] | null>(null);
  const [expanded, setExpanded] = useState(false);
  const flaggedSet = new Set(flagged);
  const axis = spaghettiAxisLabels(data);

  const W = 760, H = 220, ml = 52, mr = 12, mt = 12, mb = 38;
  const allYAll = data.series.flatMap(s => s.y).filter(v => v > 0);
  const allXAll = data.series.flatMap(s => s.x).filter(isFinite);
  if (!allYAll.length || !allXAll.length) return <div style={{ fontSize: 12, color: 'var(--text-dim)' }}>No data to plot.</div>;

  // x-range from the DATA, not from 0: a steady-state window sampled at
  // 168–192 h must not be squashed into the last 12% of a 0–192 h axis.
  const dataLo = Math.min(...allXAll), dataHi = Math.max(...allXAll);
  const xlo = xr ? Math.max(dataLo, Math.min(xr[0], xr[1])) : dataLo;
  const xhi = xr ? Math.min(dataHi, Math.max(xr[0], xr[1])) : dataHi;
  const inRange = (x: number) => x >= xlo - 1e-9 && x <= xhi + 1e-9;
  const allY = data.series.flatMap(s => s.y.filter((v, j) => v > 0 && inRange(s.x[j])));
  if (!allY.length) return <div>{/* nothing inside the window */}
    <div style={{ fontSize: 12, color: 'var(--text-dim)' }}>No observations in {fmt(xlo, 1)}–{fmt(xhi, 1)}.</div>
    <button type="button" className="link-btn" style={{ marginTop: 4 }} onClick={() => setXr(null)}>Reset time window</button>
  </div>;
  const xpad = (xhi - xlo || xhi || 1) * 0.04;
  const xmin = xlo - xpad < 0 && xlo >= 0 ? 0 : xlo - xpad;
  const xmax = xhi + xpad;
  const ymin = Math.min(...allY), ymax = Math.max(...allY) * 1.1;
  const cw = W - ml - mr, ch = H - mt - mb;
  const sx = (x: number) => ml + ((x - xmin) / (xmax - xmin || 1)) * cw;
  const lmin10 = Math.log10(ymin * 0.8), lmax10 = Math.log10(ymax);
  const syLog = (y: number) => y > 0 ? H - mb - (Math.log10(y) - lmin10) / (lmax10 - lmin10) * ch : H - mb;
  const syLin = (y: number) => H - mb - (y / ymax) * ch;
  const sy = logY ? syLog : syLin;

  // Series colour: with a flagged subject present, the flagged one is red and
  // the rest recede to the accent (as in the comp); otherwise the per-subject
  // palette so individual curves can still be told apart.
  const anyFlagged = data.series.some(s => flaggedSet.has(String(s.id)));
  const seriesColor = (id: string, i: number) =>
    flaggedSet.has(id) ? 'var(--red)' : anyFlagged ? 'var(--accent)' : SPAG_PALETTE[i % SPAG_PALETTE.length];
  const seriesOpacity = (id: string) => (flaggedSet.has(id) ? 0.9 : anyFlagged ? 0.5 : 0.7);

  const caption = `${data.n_subjects} subjects`
    + (data.blq_excluded > 0 ? ` · ${data.blq_excluded} points below the LLOQ excluded` : '')
    + flaggedSentence(flagged.filter(id => data.series.some(s => String(s.id) === id)));

  const controls = (
    <div className="chart-controls">
      <div className="seg" role="group" aria-label="Y axis scale">
        <button type="button" aria-pressed={logY} onClick={() => setLogY(true)}>Log</button>
        <button type="button" aria-pressed={!logY} onClick={() => setLogY(false)}>Linear</button>
      </div>
      <div className="seg" role="group" aria-label="View">
        <button type="button" aria-pressed={!individual} onClick={() => setIndividual(false)}>Overlay</button>
        <button type="button" aria-pressed={individual} onClick={() => setIndividual(true)}>Individual</button>
      </div>
      <XRangeSlider min={dataLo} max={dataHi} value={[xlo, xhi]} label="Time window"
        onChange={v => setXr(v)} onReset={() => setXr(null)} />
      {!individual && (
        <button type="button" className="chart-expand" aria-label="Open concentration–time chart full screen"
          title="Open full screen — wheel to zoom, drag to pan" onClick={() => setExpanded(true)}>
          <Maximize2 size={13} aria-hidden="true" />
        </button>
      )}
    </div>
  );

  if (individual) {
    const sw = 210, sh = 150, sml = 42, smr = 8, smt = 8, smb = 28;
    const xTicksI = niceTicks(xmin, xmax, 3).filter(v => v >= xmin && v <= xmax);
    const scx = (x: number) => sml + ((x - xmin) / (xmax - xmin || 1)) * (sw - sml - smr);
    const tickLabel = (v: number) => v >= 1000 ? `${(v / 1000).toFixed(0)}k` : v >= 1 ? String(Math.round(v)) : v.toPrecision(1);
    return (
      <figure className="chart-figure">
        {controls}
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6 }}>
          {data.series.map((s, i) => {
            const id = String(s.id);
            const color = seriesColor(id, i);
            const pts = s.x.map((x, j) => ({ x, y: s.y[j] })).filter(p => p.y > 0 && inRange(p.x));
            if (!pts.length) return null;
            const ays = pts.map(p => p.y);
            const yTop = Math.max(...ays) * 1.1, yBot = Math.min(...ays) * 0.8;
            const aln = Math.log10(yBot), alx = Math.log10(yTop);
            const ach = sh - smt - smb;
            const scy = (y: number) => logY
              ? (y > 0 ? sh - smb - (Math.log10(y) - aln) / (alx - aln || 1) * ach : sh - smb)
              : sh - smb - (y / yTop) * ach;
            // y ticks: decades inside the panel's range (log) or quarter steps (linear);
            // fall back to the panel's min/max so every panel shows at least two labels
            let yTicksI: number[] = logY
              ? Array.from({ length: Math.ceil(alx) - Math.floor(aln) + 1 }, (_, k) => Math.pow(10, Math.floor(aln) + k))
                .filter(v => v >= yBot && v <= yTop)
              : [0.5, 1.0].map(f => f * yTop / 1.1);
            if (yTicksI.length < 2) yTicksI = [Math.min(...ays), Math.max(...ays)];
            const poly = pts.map(p => `${scx(p.x).toFixed(1)},${scy(p.y).toFixed(1)}`).join(' ');
            return (
              <Zoomable key={s.id} title={`${s.id} — ${axis.y} vs ${axis.x}`}>
              <svg viewBox={`0 0 ${sw} ${sh}`} width={sw} role="img"
                aria-label={`Subject ${s.id}: ${axis.y} vs ${axis.x}${flaggedSet.has(id) ? ', terminal phase flagged' : ''}`}
                style={{ background: 'rgba(31,102,166,0.03)', borderRadius: 4, border: `1px solid ${flaggedSet.has(id) ? 'var(--red)' : 'var(--border)'}`, display: 'block' }}>
                <line x1={sml} y1={sh - smb} x2={sw - smr} y2={sh - smb} stroke="var(--border)" />
                <line x1={sml} y1={smt} x2={sml} y2={sh - smb} stroke="var(--border)" />
                {yTicksI.map((v, k) => (
                  <g key={k}>
                    <line x1={sml - 3} y1={scy(v)} x2={sml} y2={scy(v)} stroke="var(--text-dim)" />
                    <text x={sml - 4} y={scy(v) + 3} textAnchor="end" fontSize="9" {...TICK_TEXT}>{tickLabel(v)}</text>
                  </g>
                ))}
                {xTicksI.map((v, k) => (
                  <g key={k}>
                    <line x1={scx(v)} y1={sh - smb} x2={scx(v)} y2={sh - smb + 3} stroke="var(--text-dim)" />
                    <text x={scx(v)} y={sh - smb + 12} textAnchor="middle" fontSize="9" {...TICK_TEXT}>{v}</text>
                  </g>
                ))}
                {pts.length > 1 && <polyline points={poly} fill="none" stroke={color} strokeWidth="1.3" strokeOpacity="0.8" />}
                {pts.map((p, j) => <circle key={j} cx={scx(p.x)} cy={scy(p.y)} r="2" fill={color} />)}
                <text x={(sml + sw - smr) / 2} y={sh - 3} textAnchor="middle" fontSize="10" fontWeight="600" fill="var(--text)">{s.id}</text>
                <text x={10} y={(smt + sh - smb) / 2} textAnchor="middle" fontSize="9" fill="var(--text-dim)"
                  transform={`rotate(-90 10 ${(smt + sh - smb) / 2})`}>{logY ? 'log' : 'linear'}</text>
              </svg>
              </Zoomable>
            );
          })}
        </div>
        <div style={{ fontSize: 11, color: 'var(--text)', marginTop: 6 }}>
          x: <b>{axis.x}</b> {fmt(xlo, 1)}–{fmt(xhi, 1)} (shared) · y: <b>{axis.y}</b>{logY ? ' (log scale)' : ''}, scaled per subject
        </div>
        <figcaption className="figcap">{caption}</figcaption>
      </figure>
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

  const xTicks = niceTicks(xmin, xmax, 6).filter(v => v >= xmin && v <= xmax);

  // Flagged series are drawn last so they sit on top of the others.
  const ordered = data.series.map((s, i) => ({ s, i }))
    .sort((a, b) => Number(flaggedSet.has(String(a.s.id))) - Number(flaggedSet.has(String(b.s.id))));

  const overlaySvg = (
      <svg viewBox={`0 0 ${W} ${H}`} style={{ width: '100%', display: 'block' }} role="img"
        aria-label={`${logY ? 'Log-scale' : 'Linear'} ${axis.y} vs ${axis.x} for ${data.n_subjects} subjects`}>
        <line x1={ml} y1={H - mb} x2={W - mr} y2={H - mb} stroke="var(--border)" />
        <line x1={ml} y1={mt} x2={ml} y2={H - mb} stroke="var(--border)" />
        {yTicks.map((v, k) => {
          const yy = sy(v);
          const lbl = v >= 1000 ? `${(v / 1000).toFixed(0)}k` : v >= 1 ? String(Math.round(v)) : v.toPrecision(1);
          return (
            <g key={k}>
              <line x1={ml - 3} y1={yy} x2={ml} y2={yy} stroke="var(--text-dim)" />
              <text x={ml - 5} y={yy + 3.5} textAnchor="end" fontSize="10" {...TICK_TEXT}>{lbl}</text>
            </g>
          );
        })}
        {xTicks.map((v, k) => (
          <g key={k}>
            <line x1={sx(v)} y1={H - mb} x2={sx(v)} y2={H - mb + 3} stroke="var(--text-dim)" />
            <text x={sx(v)} y={H - mb + 13} textAnchor="middle" fontSize="10" {...TICK_TEXT}>{v}</text>
          </g>
        ))}
        <text x={(ml + W - mr) / 2} y={H - 6} textAnchor="middle" {...AXIS_TITLE}>{axis.x}</text>
        <text x={14} y={(mt + H - mb) / 2} textAnchor="middle" {...AXIS_TITLE}
          transform={`rotate(-90 14 ${(mt + H - mb) / 2})`}>{axis.y}</text>
        {ordered.map(({ s, i }) => {
          const id = String(s.id);
          const isFlagged = flaggedSet.has(id);
          const color = seriesColor(id, i);
          const pts = s.x.map((x, j) => ({ x, y: s.y[j] })).filter(p => p.y > 0 && inRange(p.x));
          if (!pts.length) return null;
          const poly = pts.map(p => `${sx(p.x).toFixed(1)},${sy(p.y).toFixed(1)}`).join(' ');
          const last = pts[pts.length - 1];
          return (
            <g key={s.id}>
              {pts.length > 1 && <polyline points={poly} fill="none" stroke={color}
                strokeWidth={isFlagged ? '1.5' : '1.2'} strokeOpacity={seriesOpacity(id)} />}
              {pts.map((p, j) => <circle key={j} cx={sx(p.x)} cy={sy(p.y)} r={isFlagged ? 2.5 : 2}
                fill={color} fillOpacity={isFlagged ? 0.9 : 0.85} />)}
              {isFlagged && (
                <text x={Math.min(sx(last.x), W - mr) - 4} y={sy(last.y) - 5} textAnchor="end"
                  fontSize="11" fill="var(--red)">subject {id}</text>
              )}
            </g>
          );
        })}
      </svg>
  );

  return (
    <figure className="chart-figure">
      {controls}
      {overlaySvg}
      {expanded && <ZoomModal title="Concentration–time" onClose={() => setExpanded(false)}>{overlaySvg}</ZoomModal>}
      <figcaption className="figcap">{caption}</figcaption>
    </figure>
  );
}

interface LzManualFit {
  lambda_z: number; lambda_z_intercept: number;
  t_half: number; r2_adj: number; n_pts: number;
  lz_x: number[]; lz_y: number[];
  fit_x: number[]; fit_y: number[];
}

/** Panels shown before "All N subjects". */
const LZ_PANELS_COLLAPSED = 4;
/** Radius of the transparent hit circle around each λz point: 12 viewBox
 *  units = 24px at the panel's 1:1 rendering, the minimum touch target. Per
 *  panel it is clamped to half the smallest x-gap between neighbouring points
 *  (never below LZ_HIT_R_MIN) so hit circles cannot cover a neighbour's dot —
 *  the later sibling would otherwise win the click and toggle the wrong point. */
const LZ_HIT_R = 12;
const LZ_HIT_R_MIN = 4;

function NcaLzPlot({ data, sessionId, flagged = {}, hours = false }: {
  data: NcaPlotData; sessionId: string;
  /** subject id → %AUC extrapolated, for subjects over the 20 % limit (from the NCA table). */
  flagged?: Record<string, number>;
  /** true when the dataset's time axis is labelled in hours — the only case an 'h' is printed. */
  hours?: boolean;
}) {
  const subjects = useMemo(() => bySubjectId(data.subjects, s => s.id), [data.subjects]);

  const [selections, setSelections] = useState<Record<string, Set<string>>>(() => {
    const s: Record<string, Set<string>> = {};
    for (const sub of subjects) s[sub.id] = new Set(sub.lz_x.map(v => v.toFixed(4)));
    return s;
  });
  const [localFits, setLocalFits] = useState<Record<string, LzManualFit>>({});
  const [loadingId, setLoadingId] = useState<string | null>(null);
  const [errors, setErrors] = useState<Record<string, string>>({});
  const [showAll, setShowAll] = useState(false);

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
  const sw = 164, sh = 84, sml = 34, smr = 6, smt = 10, smb = 20;
  const h = hours ? ' h' : '';
  const collapsible = subjects.length > LZ_PANELS_COLLAPSED;
  const visible = collapsible && !showAll ? subjects.slice(0, LZ_PANELS_COLLAPSED) : subjects;
  // One sentence per flagged subject, from the NCA row (n_pts = points in the
  // current λz fit; the %AUC figure is the NCA table's own value).
  const flaggedSentences = subjects
    .filter(s => flagged[s.id] != null)
    .map(s => ` Subject ${s.id} uses ${s.n_pts ?? '?'} late points and extrapolates ${fmt(flagged[s.id], 1)} % of its AUC.`)
    .join('');

  return (
    <div className="lz-block">
      <div className="lz-head">
        <p className="lz-intro">Click points to include or exclude them, then refit.{flaggedSentences}</p>
        {collapsible && (
          <button type="button" className="link-btn" aria-expanded={showAll} onClick={() => setShowAll(v => !v)}>
            {showAll ? `First ${LZ_PANELS_COLLAPSED} subjects` : `All ${subjects.length} subjects`}
          </button>
        )}
      </div>
      <div className="lz-grid">
        {visible.map((s: LzSubject) => {
          const sel = selections[s.id] ?? new Set<string>();
          const fit = localFits[s.id];
          const isLoading = loadingId === s.id;
          const isFlagged = flagged[s.id] != null;

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

          // Hit radius for this panel: no circle may reach a neighbour's centre.
          const obsPx = allObs.map(p => scx(p.x)).sort((a, b) => a - b);
          const minGap = obsPx.slice(1).reduce((m, x, k) => Math.min(m, x - obsPx[k]), Infinity);
          const hitR = Math.max(LZ_HIT_R_MIN, Math.min(LZ_HIT_R, minGap / 2));

          const nSel = sel.size;
          const canRefit = nSel >= 3;
          const tHalfText = activeTHalf != null ? `t½ ${activeTHalf.toFixed(1)}${h}` : null;
          const svgLabel = `Subject ${s.id} terminal slope, ${activeN ?? '?'} points`
            + (activeTHalf != null ? `, half-life ${activeTHalf.toFixed(1)}${hours ? ' hours' : ''}` : '')
            + (isFlagged ? `, extrapolation ${fmt(flagged[s.id], 1)} percent, over the limit` : '');

          return (
            <figure key={s.id} className={`lz-panel ${isFlagged ? 'flagged' : ''}`}>
              <svg viewBox={`0 0 ${sw} ${sh}`} width={sw} role="group" aria-label={svgLabel} className="lz-svg">
                <line x1={sml} y1={sh - smb} x2={sw - smr} y2={sh - smb} stroke="var(--border)" />
                <line x1={sml} y1={smt} x2={sml} y2={sh - smb} stroke="var(--border)" />
                {yTicks.map((v, k) => {
                  const yy = scy(v);
                  return (
                    <g key={k}>
                      <line x1={sml - 3} y1={yy} x2={sml} y2={yy} stroke="var(--border)" />
                      <text x={sml - 4} y={yy + 3} textAnchor="end" fontSize="9" {...TICK_TEXT}>
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
                      <text x={scx(v)} y={sh - smb + 12} textAnchor="middle" fontSize="9" {...TICK_TEXT}>{v}</text>
                    </g>
                  ) : null;
                })}
                {activeFitX.length > 1 && (
                  <polyline points={fitPoly} fill="none" stroke={isFlagged ? 'var(--yellow)' : 'var(--text-dim)'}
                    strokeWidth="1.4" strokeDasharray="4 3" />
                )}
                {allObs.map((p, j) => {
                  const tk = p.x.toFixed(4);
                  const isSelected = sel.has(tk);
                  const cx = scx(p.x), cy = scy(p.y);
                  return (
                    <g key={j}>
                      {/* transparent hit target (≤24px, clamped so neighbours never overlap), keyboard-reachable; the visible dot sits on top */}
                      <circle className="lz-hit" cx={cx} cy={cy} r={hitR} fill="transparent"
                        tabIndex={0} role="button" aria-pressed={isSelected}
                        aria-label={`t = ${p.x}: ${isSelected ? 'included in λz' : 'excluded from λz'}`}
                        onClick={() => toggle(s.id, tk)}
                        onKeyDown={e => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); toggle(s.id, tk); } }} />
                      <circle cx={cx} cy={cy} r={isSelected ? 3 : 2.5} pointerEvents="none"
                        fill={isSelected ? 'var(--accent)' : 'var(--border-strong)'} />
                    </g>
                  );
                })}
              </svg>
              <figcaption className="lz-cap">
                <span className="lz-cap-row lz-cap-main">
                  <b>{s.id}</b>{tHalfText && <> · {tHalfText}</>} · {activeN ?? '?'} pts
                </span>
                <span className="lz-cap-row">
                  <span>
                    {activeR2 != null ? <>R² {activeR2.toFixed(3)}</> : 'R² –'}
                    {nSel !== activeN && <> · {nSel} selected</>}
                    {fit && <> · manual</>}
                  </span>
                  <span className="lz-links">
                    <button type="button" className="link-btn" onClick={() => doRefit(s)} disabled={!canRefit || isLoading}
                      title={canRefit ? 'Refit λz with the selected points' : 'Select at least 3 points'}>
                      {isLoading ? 'Refitting…' : 'Refit'}
                    </button>
                    <button type="button" className="link-btn" onClick={() => reset(s)} disabled={isLoading}>Reset</button>
                  </span>
                </span>
              </figcaption>
              {errors[s.id] && <div className="lz-error" role="alert">{errors[s.id]}</div>}
            </figure>
          );
        })}
      </div>
      {/* swatches are empty coloured spans, not glyphs, so no low-contrast "text" is rendered */}
      <div className="lz-legend">
        <span><span className="lz-swatch dot" style={{ background: 'var(--accent)' }} aria-hidden="true" /> Included in λz</span>
        <span><span className="lz-swatch dot" style={{ background: 'var(--border-strong)' }} aria-hidden="true" /> Excluded</span>
        <span><span className="lz-swatch dash" aria-hidden="true" /> Regression fit (amber when flagged)</span>
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
                  disabled={loading} aria-label={`Role for column ${c.name}`}
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
  const [auditIntegrity, setAuditIntegrity] = useState<AuditIntegrityStatus | null>(null);
  const [healthy, setHealthy] = useState<boolean | null>(null);
  /** /api/health `auth`: null until known. Drives the gate's attribution copy. */
  const [authMode, setAuthMode] = useState<'required' | 'open' | null>(null);
  const [llmLabel, setLlmLabel] = useState<string>('');
  const [llmOpen, setLlmOpen] = useState(false);
  /** Client-side gate wait only: 'pending' from the moment a gate arrives
   *  until the next run (or a rejection) clears it. Approved / Rejected are
   *  never held here — `decisionView` below reads them from the sealed
   *  human_review audit entry, so nothing can show a decision the backend
   *  did not record. */
  const [decision, setDecision] = useState<Decision>(null);
  /** A gate decision is in flight (its POST unanswered): the gate stays up
   *  with its reason, and the review pill shows progress instead. */
  const [deciding, setDeciding] = useState(false);
  const [verifyNote, setVerifyNote] = useState<VerifyNote | null>(null);
  /** ≤1180px the review column is a drawer; this is its open state. */
  const [reviewOpen, setReviewOpen] = useState(false);
  const [calcOpen, setCalcOpen] = useState(false);
  const [currentStep, setCurrentStep] = useState(-1);
  /** Index of the step the workflow is paused at (from the gate payload). */
  const [gateStep, setGateStep] = useState(-1);
  /** Executed workflow steps that put no section in the document (load,
   *  validate, adversarial review…): shown once, folded, as the run log. */
  const [runLog, setRunLog] = useState<WorkflowExecutedStep[]>([]);
  /** Id of the upload-time "Dataset loaded" note; dropped once the workflow's
   *  own load_dataset step has reported the same fact. */
  const loadedNoteId = useRef('');
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
  /** Actions popover above the composer (quick-action rows live inside it). */
  const [actionsOpen, setActionsOpen] = useState(false);
  const actionsRef = useRef<HTMLButtonElement>(null);
  const messagesEnd = useRef<HTMLDivElement>(null);
  const fileRef = useRef<HTMLInputElement>(null);
  const shownMarkers = useRef(new Set<string>());

  const scrollBottom = () => messagesEnd.current?.scrollIntoView({ behavior: 'smooth' });
  useEffect(() => { scrollBottom(); }, [messages]);

  useEffect(() => {
    api.health()
      .then(h => { setHealthy(h.status === 'ok'); setLlmLabel(h.llm ?? ''); setAuthMode(h.auth ?? null); })
      .catch(() => setHealthy(false));
    api.createSession()
      .then(s => setSession(s))
      .catch(console.error);
    api.listPkModels()
      .then(r => setPkModels(r.models))
      .catch(() => { /* library unavailable */ });
  }, []);

  // Accept an optional `id` (most call sites pass `id: ''` as a placeholder) and
  // assign a fresh unique id when none is given, so React keys stay stable and
  // distinct; a caller passes its own (newMsgId) only to remove the row later.
  /** Push order of the last message (see DisplayMsg.seq); read synchronously
   *  at pushMsg time, before React commits the push. */
  const msgSeq = useRef(0);
  const pushMsg = useCallback((m: Omit<DisplayMsg, 'id'> & { id?: string }) => {
    const seq = ++msgSeq.current;
    setMessages(prev => [...prev, { ...m, id: m.id || newMsgId(), seq }]);
  }, []);

  function extractMessages(raw: ChatMessage[], agent: string): DisplayMsg[] {
    const out: DisplayMsg[] = [];
    for (const m of raw) {
      // Backend agents emit bare strings (AgentResult.messages: list[str]);
      // render each as an assistant reply attributed to the routing agent.
      if (typeof m === 'string') {
        if (m.trim()) out.push({ role: 'assistant', content: m, agent, id: '' });
      } else if (typeof m.content === 'string') {
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
      setAuditIntegrity(a.integrity);
    } catch { /* best-effort */ }
  }

  // Audit freshness: seals and step timings follow every state change (chat,
  // tool and calculator responses), not only workflow gates. Async then-branch
  // only; the `alive` flag drops a stale response after unmount/re-run. This
  // fetch is sent after the response that changed `state` landed, so it is
  // the one that binds the cards pushed with that response (bindSeals);
  // `seqAtFetch` keeps it from binding a card pushed after it was sent.
  useEffect(() => {
    if (!session || !state) return;
    let alive = true;
    const seqAtFetch = msgSeq.current;
    api.getAudit(session.id).then(r => {
      if (!alive) return;
      setAudit(r.entries);
      setAuditIntegrity(r.integrity);
      setMessages(prev => bindSeals(prev, r.entries, seqAtFetch));
      setVerifyNote(null);
    }).catch(() => { /* best-effort */ });
    return () => { alive = false; };
  }, [session, state]);

  /** Explicit chain check from the audit trail: re-reads the chain and reports
   *  the backend's own integrity verdict beside the link. */
  async function verifyChain() {
    if (!session) return;
    try {
      const r = await api.getAudit(session.id);
      setAudit(r.entries);
      setAuditIntegrity(r.integrity);
      const { mode, chain_ok, verified } = r.integrity;
      if (verified) {
        setVerifyNote({ tone: 'ok', text: 'Chain verified' });
      } else if (mode === 'hash_only' && chain_ok) {
        // Hash-only (dev) mode has no MAC or anchor: structure is all we can say.
        setVerifyNote({ tone: 'dim', text: 'Chain structure intact (unauthenticated)' });
      } else {
        const failed = integrityFailures(r.integrity);
        setVerifyNote({
          tone: 'bad',
          text: failed.length ? `Chain not verified: ${failed.join(', ')} failed` : 'Chain not verified',
        });
      }
    } catch {
      setVerifyNote({ tone: 'bad', text: 'Chain not verified' });
    }
  }

  /** v1: a fresh session is a page reload — App holds 40+ state slots plus the
   *  `shownMarkers` ref, so an in-place reset is a follow-up. */
  function newSession() {
    window.location.reload();
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

  /** The server decides whether a workflow leg runs inline or as a job (any real
   *  population fit goes to the queue). Accept either shape: poll a job handle,
   *  then render the WorkflowResponse. Fixes "Cannot read properties of
   *  undefined (reading 'current_step')" on Run Modeling + Engines. */
  async function settleWorkflow(res: WorkflowStartResponse, note = 'Population fit running…') {
    if ('job_id' in res && res.job_id) {
      if (!session) return;
      handleWorkflowResponse(await pollWorkflowJob(session.id, res.job_id, note));
      return;
    }
    handleWorkflowResponse(res as WorkflowResponse);
  }

  /** Poll a workflow leg the backend queued. The audit trail is re-read on
   *  the first tick and periodically after it, so the seals the leg adds —
   *  the human_review entry first — reach the rail and the review pill while
   *  the leg runs: both read the sealed decision, never the click. */
  async function pollWorkflowJob(sid: string, jobId: string, note: string): Promise<WorkflowResponse> {
    let ticks = 0;
    try {
      return await api.pollJob<WorkflowResponse>(sid, jobId, s => {
        setJobNote(`${note} ${s}s (several real fits — this can take minutes)`);
        ticks += 1;
        if (ticks === 1 || ticks % AUDIT_REFRESH_EVERY_TICKS === 0) void refreshAudit();
      });
    } finally {
      setJobNote('');
    }
  }

  /** A leg the backend accepted lost its client-side tracking — the job poll
   *  failed after its retries, or the job itself reported an error. The
   *  backend session is the truth: reload its state and audit so the document
   *  and the trail show what was actually sealed, then settle the status from
   *  that state — complete when the workflow's last step ran, otherwise idle
   *  so the run can be repeated. Never a dead 'error'. */
  async function reconcileRun(err: Error, workflow: WorkflowName) {
    pushMsg({
      role: 'assistant', agent: 'supervisor', id: '',
      content: `Error: ${err.message} — session state reloaded; the audit trail shows what was sealed.`,
    });
    if (!session) return;
    let finished = false;
    try {
      const st = await api.getState(session.id);
      setState(st);
      setCurrentStep(st.current_step ?? -1);
      finished = st.workflow_name === workflow
        && (st.current_step ?? -1) >= WORKFLOW_UI[workflow].steps.length;
    } catch { /* the audit refresh below still reconciles the trail */ }
    await refreshAudit();
    setWfStatus(finished ? 'complete' : 'idle');
  }

  function handleWorkflowResponse(res: WorkflowResponse) {
    setState(res.state);
    setCurrentStep(res.state.current_step ?? -1);

    if (res.messages) {
      const agent = res.state.last_agent ?? 'supervisor';
      extractMessages(res.messages, agent).forEach(m => pushMsg(m));
    }

    // Backend tools whose result section this leg puts in the document. A step
    // that has one is not narrated as well — the section's attribution line
    // already names agent, tool and seal; the rest go to the folded run log.
    const sectioned = new Set<string>();
    if (res.state.dataset_metadata) sectioned.add('profile_pk_dataset');
    const pushCard = (marker: Marker, agent: string) => {
      pushMsg({ role: 'assistant', content: marker, agent, id: '', snap: res.state });
      const tool = CARD_META[marker].tool;
      if (tool) sectioned.add(tool);
    };

    // One result card per step the leg ran. `executed` is the only per-step
    // record a workflow leg returns — a leg polled from a job (NLME → SCM →
    // diagnostics → forest → VPC) arrives here exactly like an inline one, so
    // this is what keeps the document complete. Every card below is pushed
    // once per run: `shownMarkers` is cleared by uploadAndRun and each template
    // tool runs once, so a resumed leg (gate approve/reject) must not re-emit a
    // section the page already shows. The key is the marker, not the audit
    // seal — `audit` is refreshed after these pushes, so a seal is not known here.
    const executed = res.executed ?? [];
    for (const step of executed) {
      const agent = step.agent || res.state.last_agent || 'supervisor';
      const card = STEP_CARD[step.tool];
      if (card && res.state[card.key] && !shownMarkers.current.has(card.marker)) {
        shownMarkers.current.add(card.marker);
        pushCard(card.marker, agent);
      }
    }

    if (res.state.spaghetti_data && !shownMarkers.current.has('spaghetti')) {
      shownMarkers.current.add('spaghetti');
      pushCard('__SPAGHETTI__', 'data_manager');
    }
    if (res.state.nca_summary && !shownMarkers.current.has('nca')) {
      shownMarkers.current.add('nca');
      pushCard('__NCA_TABLE__', 'nca');
    }
    if (res.state.nca_plot_data && !shownMarkers.current.has('nca_lz')) {
      shownMarkers.current.add('nca_lz');
      pushCard('__NCA_LZ__', 'nca');
    }
    if (res.state.pk_model_results?.status === 'ok' && !shownMarkers.current.has('pkmodel')) {
      shownMarkers.current.add('pkmodel');
      pushCard('__PKMODEL__', 'modeler');
    }
    if (res.state.engine_comparison_results && !shownMarkers.current.has('engines')) {
      shownMarkers.current.add('engines');
      pushCard('__ENGINES__', 'modeler');
    }
    if (res.state.qc_verdict && !shownMarkers.current.has('qc')) {
      shownMarkers.current.add('qc');
      pushCard('__QC_CARD__', 'qc');
    }

    // Steps with no section of their own are logged, not narrated; the upload
    // note is dropped once load_dataset has reported the same fact.
    const unsectioned = executed.filter(s => !sectioned.has(s.tool));
    if (unsectioned.length) setRunLog(prev => [...prev, ...unsectioned]);
    if (loadedNoteId.current && executed.some(s => s.tool === 'load_dataset')) {
      const noteId = loadedNoteId.current;
      loadedNoteId.current = '';
      setMessages(prev => prev.filter(m => m.id !== noteId));
    }

    if (res.status === 'awaiting_review') {
      setGateStep(res.review?.after_step ?? (res.state.current_step ?? 0) - 1);
      setDecision('pending');
      setWfStatus('awaiting_review');
      // ≤1180px the panel is a closed drawer: a gate opens it, so a paused run
      // is never signalled by the topbar toggle alone (ReviewPanel then focuses
      // its heading in the same commit).
      if (window.matchMedia(REVIEW_DRAWER_MQ).matches) setReviewOpen(true);
      refreshAudit();
    } else if (res.status === 'complete') {
      setWfStatus('complete');
      // Same drawer rule: the report / complete card lives only in the panel
      // (chat suppresses __REPORT__), so a narrow viewport must open it.
      if (window.matchMedia(REVIEW_DRAWER_MQ).matches) setReviewOpen(true);
      if (res.state.report_path) {
        pushMsg({ role: 'assistant', content: '__REPORT__', agent: 'report', id: '' });
      }
      refreshAudit();
    } else if (res.status === 'rejected') {
      // Still maps to idle (re-run possible). The rejection is not held
      // client-side: the refreshed audit's sealed human_review entry is what
      // the rail and the review panel read (`decisionView`).
      setWfStatus('idle');
      setDecision(null);
      refreshAudit();
    } else {
      setWfStatus('idle');
    }
  }

  async function uploadAndRun(workflow: WorkflowName = 'nca_full') {
    if (!session || !file) return;
    shownMarkers.current.clear();
    setRunLog([]);
    loadedNoteId.current = '';
    setDecision(null);
    // A new run has no gate yet: an earlier run's sealed decision at the same
    // step must not read as this run's.
    setGateStep(-1);
    setActiveWorkflow(workflow);
    setLoading(true);
    setWfStatus('running');
    setCurrentStep(0);
    // No chat echo of the start: the document header already names the run.
    let accepted = false;
    try {
      const up = await api.uploadDataset(session.id, file);
      const meta = up.metadata;
      loadedNoteId.current = newMsgId();
      pushMsg({
        role: 'assistant',
        content: `Dataset loaded: ${meta['n_records']} records, ${meta['n_subjects']} subjects, ${meta['n_columns']} columns.`,
        agent: 'data_manager',
        id: loadedNoteId.current,
      });
      const res = await api.startWorkflow(session.id, meta['dataset_path'] as string ?? '', workflow);
      accepted = true;
      await settleWorkflow(res);
    } catch (e) {
      if (accepted) {
        await reconcileRun(e as Error, workflow);
      } else {
        // Nothing ran (a bad CSV, a refused start): back to idle, so the file
        // can be replaced and the run repeated without a reload.
        pushMsg({ role: 'assistant', content: `Error: ${(e as Error).message}`, agent: 'supervisor', id: '' });
        setWfStatus('idle');
      }
    } finally {
      setJobNote('');
      setLoading(false);
    }
  }

  async function resume(approve: boolean, reason = '') {
    if (!session) return;
    setLoading(true);
    setDeciding(true);
    // The gate stays exactly as it is — wfStatus 'awaiting_review', decision
    // 'pending', the decision card and its reason mounted — until the backend
    // has taken the decision. A request that fails therefore leaves the gate
    // open to decide again, and nothing can read as Approved or Rejected
    // before the human_review entry exists (`decisionView` reads the seal).
    // Approving runs every remaining step in one call. If any of them is a real
    // population fit, poll a job instead of holding the request open — and say
    // what actually happens next rather than assuming the NCA shape.
    const remaining = WORKFLOW_UI[activeWorkflow].steps.slice(Math.max(currentStep, 0));
    const isLongLeg = approve && remaining.some(s => HEAVY_STEPS.has(s.key));
    const note = !approve ? 'Rejected — workflow stopped.'
      : isLongLeg ? 'Approved — running the population fit (NLME → SCM → diagnostics → forest → VPC).'
      : 'Approved — generating report.';
    pushMsg({ role: 'user', content: note, id: '' });
    let accepted = false;
    try {
      if (isLongLeg) {
        const { job_id } = await api.resumeWorkflowAsync(session.id, reason);
        accepted = true;
        setWfStatus('running');
        handleWorkflowResponse(await pollWorkflowJob(session.id, job_id, 'Population fit running…'));
        return;
      }
      const res = await api.resumeWorkflow(session.id, approve, reason);
      accepted = true;
      setWfStatus('running');
      await settleWorkflow(res);
    } catch (e) {
      if (accepted) {
        await reconcileRun(e as Error, activeWorkflow);
      } else {
        pushMsg({
          role: 'assistant', agent: 'supervisor', id: '',
          content: `Error: ${(e as Error).message} — the decision was not recorded; the gate is still open, decide again to retry.`,
        });
      }
    } finally {
      setDeciding(false);
      setJobNote('');
      setLoading(false);
    }
  }

  const AGENT_CARD: Record<string, string> = {
    be: '__BE__', dose_prop: '__DP__',
    compartmental: '__COMPARTMENTAL__', poppk: '__POPPK__', statistician: '__STATS__', clinpharm: '__CLINPHARM__',
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
      // An expensive tool the agent PROPOSED (e.g. run_simest): nothing ran —
      // render the approve/reject card bound to this snapshot's proposal.
      if (res.pending_tool) {
        pushMsg({ role: 'assistant', content: '__PENDING_TOOL__', agent: res.agent, id: '', snap: res.state });
      }
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

  const PROPOSAL_CARD: Record<string, string> = {
    run_simest: '__SIMEST__', run_bootstrap: '__BOOTSTRAP__',
    run_sir: '__SIR__', run_profile: '__PROFILE__',
  };

  /** Human decision on a chat-proposed expensive tool. Approval is the tool's
   *  `confirm` and runs it as a background job (polled); rejection is inline. */
  async function decidePendingTool(approve: boolean) {
    if (!session) return;
    const tool = state?.pending_tool?.tool ?? 'proposed tool';
    setLoading(true);
    pushMsg({ role: 'user', content: `${approve ? 'Approve' : 'Reject'} ${tool}`, id: '' });
    try {
      const res = await api.decidePendingTool(session.id, approve);
      if (!approve || !res.job_id) {
        if (res.state) setState(res.state);
        pushMsg({ role: 'assistant', content: `${tool} rejected — nothing computed.`,
          agent: 'supervisor', id: '' });
        return;
      }
      // approved: the proposal is already cleared server-side; clear it locally
      // so the card stops offering buttons, then poll the job.
      setState(s => (s ? { ...s, pending_tool: null } : s));
      const job = await api.pollJob(session.id, res.job_id,
        s => setJobNote(`${tool} running as a background job… ${s}s`));
      setJobNote('');
      setState(job.state);
      pushMsg({ role: 'assistant', content: job.summary, agent: 'simulator', id: '' });
      const card = PROPOSAL_CARD[tool];
      if (card) pushMsg({ role: 'assistant', content: card, agent: 'simulator', id: '', snap: job.state });
    } catch (e) {
      pushMsg({ role: 'assistant', content: `Error: ${(e as Error).message}`, agent: 'supervisor', id: '' });
    } finally { setJobNote(''); setLoading(false); }
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

  async function runNlme(method: string, opts?: { prior_from?: string; prior_var?: number }) {
    if (!session) return;
    const label: Record<string, string> = {
      focei: 'FOCE-I only', saem: 'SAEM',
      focei_saem: 'FOCE-I (SAEM-seeded)', auto: 'Auto (escalating)',
    };
    const priorNote = opts?.prior_from
      ? ` + informative prior (MAP${opts.prior_var != null ? `, var ${opts.prior_var}` : ''})` : '';
    setLoading(true);
    pushMsg({ role: 'user', content: `NLME fit — ${label[method] ?? method} (${errorModel} error)${priorNote}`, id: '' });
    try {
      const { job_id } = await api.nlme(session.id, {
        method, error_model: errorModel,
        ...(opts?.prior_from ? { prior_from: opts.prior_from } : {}),
        ...(opts?.prior_var != null ? { prior_var: opts.prior_var } : {}),
      });
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

  async function runPriorCheck() {
    if (!session) return;
    setLoading(true);
    pushMsg({ role: 'user', content: 'Prior check (predictive band + shrinkage)', id: '' });
    try {
      const res = await api.priorCheck(session.id, { n_draws: 500 });
      setState(res.state);
      pushMsg({ role: 'assistant', content: res.summary, agent: 'modeler', id: '' });
      pushMsg({ role: 'assistant', content: '__PRIORCHECK__', agent: 'modeler', id: '', snap: res.state });
    } catch (e) {
      pushMsg({ role: 'assistant', content: `Error: ${(e as Error).message}`, agent: 'modeler', id: '' });
    } finally { setLoading(false); }
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
    pushMsg({ role: 'user', content: 'Cross-engine comparison (FOCE-I vs nlmixr2 vs Monolix)', id: '' });
    try {
      const { job_id } = await api.engineComparison(session.id,
        { engines: ['pharmagent_focei', 'nlmixr2', 'monolix'] });
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

  /** Review-panel "Download DOCX": the report already exists, so fetch it
   *  through the authenticated blob path (a bare <a href> cannot send the
   *  bearer header and 401s when auth is required). */
  async function downloadExistingReport() {
    if (!session || !state?.report_path) return;
    try {
      await api.downloadReportFile(session.id, state.report_path);
    } catch (e) {
      pushMsg({ role: 'assistant', content: `Error: ${(e as Error).message}`, agent: 'report', id: '' });
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

  async function runSpecialPop(opts?: { source?: string; stratify_by?: string | null }) {
    if (!session) return;
    setLoading(true);
    pushMsg({ role: 'user', content: 'Special-population exposure simulation', id: '' });
    try {
      const res = await api.specialPopulation(session.id, {
        dose: simDose, tau: simTau, n_doses: simNDoses,
        ...(opts?.source ? { source: opts.source } : {}),
        ...(opts?.stratify_by ? { stratify_by: opts.stratify_by } : {}),
      });
      setState(res.state);
      pushMsg({ role: 'assistant', content: res.summary, agent: 'simulator', id: '' });
      pushMsg({ role: 'assistant', content: '__SPECIALPOP__', agent: 'simulator', id: '', snap: res.state });
    } catch (e) {
      pushMsg({ role: 'assistant', content: `Error: ${(e as Error).message}`, agent: 'simulator', id: '' });
    } finally { setLoading(false); }
  }

  async function runIndividualExposures() {
    if (!session) return;
    setLoading(true);
    pushMsg({ role: 'user', content: 'Individual steady-state exposures', id: '' });
    try {
      const res = await api.individualExposures(session.id, { dose: simDose, tau: simTau, n_doses: simNDoses });
      setState(res.state);
      pushMsg({ role: 'assistant', content: res.summary, agent: 'simulator', id: '' });
      pushMsg({ role: 'assistant', content: '__INDIVEXP__', agent: 'simulator', id: '', snap: res.state });
    } catch (e) {
      pushMsg({ role: 'assistant', content: `Error: ${(e as Error).message}`, agent: 'simulator', id: '' });
    } finally { setLoading(false); }
  }

  async function runPediatric(opts?: { source?: string; wt_exponent_cl?: number | null; wt_exponent_v?: number | null }) {
    if (!session) return;
    setLoading(true);
    pushMsg({ role: 'user', content: 'Pediatric dose-finding simulation', id: '' });
    try {
      const res = await api.pediatricSimulation(session.id, {
        tau: simTau, n_doses: simNDoses,
        ...(opts?.source ? { source: opts.source } : {}),
        ...(opts?.wt_exponent_cl != null ? { wt_exponent_cl: opts.wt_exponent_cl } : {}),
        ...(opts?.wt_exponent_v != null ? { wt_exponent_v: opts.wt_exponent_v } : {}),
      });
      setState(res.state);
      pushMsg({ role: 'assistant', content: res.summary, agent: 'simulator', id: '' });
      pushMsg({ role: 'assistant', content: '__PEDIATRIC__', agent: 'simulator', id: '', snap: res.state });
    } catch (e) {
      pushMsg({ role: 'assistant', content: `Error: ${(e as Error).message}`, agent: 'simulator', id: '' });
    } finally { setLoading(false); }
  }

  const QUICK_ACTIONS = [
    { label: 'Dose proportionality', msg: 'run dose proportionality power model' },
    { label: 'Compartmental fit', msg: 'fit a one- and two-compartment model' },
    { label: 'Population PK', msg: 'population pk typical values and iiv' },
    { label: 'Bioequivalence', msg: 'run a bioequivalence assessment test vs reference' },
    { label: 'Stats advice', msg: 'how should I analyze this data: parametric or non-parametric, which test?' },
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

  const wfSteps = WORKFLOW_UI[activeWorkflow].steps;
  const timings = stepTimings(audit, wfSteps);
  const datasetMeta = (state?.dataset_metadata ?? null) as
    { n_records?: unknown; n_subjects?: unknown; n_columns?: unknown } | null;
  /** What the backend will seal as the gate actor (main.py current_owner):
   *  token:<sha256[:16]> only when auth is required AND a non-blank token is
   *  set; with auth open the token is ignored and the entry is anonymous. */
  const gateSigner: GateSigner = authMode === 'open' ? 'open'
    : authMode === 'required' && token.trim() !== '' ? 'token'
    : 'anonymous';
  const statusLabel = healthy === null ? 'connecting…'
    : healthy ? `Backend online · ${describeLlm(llmLabel)}` : 'Backend offline';

  // ── Review panel inputs (all read from live state + the sealed audit) ──
  const gateDef = wfSteps[gateStep];
  const nextDef = wfSteps[gateStep + 1];
  const gate = wfStatus === 'awaiting_review' && gateDef ? {
    title: `Approve to ${nextDef ? (GATE_NEXT[nextDef.key] ?? `run "${nextDef.label}"`) : 'finish the workflow'}`,
    subtitle: gateSubtitle(activeWorkflow, gateStep),
    stepLabel: gateDef.label,
    placeholder: GATE_PLACEHOLDER[gateDef.key] ?? 'e.g. Evidence reviewed; accepted',
  } : null;
  /** What the rail and the review panel show for the gate: Approved / Rejected
   *  only from the sealed human_review entry for this run's gate; otherwise
   *  the client-side wait ('pending' at an open gate, and while the taken
   *  decision's leg runs). */
  const decisionView: Decision = sealedDecision(audit, wfSteps, gateStep) ?? decision;
  const longLeg = wfSteps.slice(Math.max(currentStep, 0)).some(s => HEAVY_STEPS.has(s.key));
  /** QC evidence belongs to the workflow that runs QC (nca_full's qc_review
   *  gate) — and to a chat "qc" turn, whose default workflow is that one. The
   *  backend never clears qc_verdict, so under a population workflow it is an
   *  earlier run's and must not shadow that run's gate evidence (fit_pk_model /
   *  adversarial_review below). */
  const runsQc = wfSteps.some(s => s.key === 'qc_review');
  const qc = runsQc && state?.qc_verdict ? {
    verdict: state.qc_verdict,
    checklist: state.qc_checklist,
    issues: state.qc_issues,
    hash: latestEntryFor(audit, 'run_qc')?.entry_hash ?? null,
    subjects: state.nca_parameters,
  } : null;
  /** Jump to the latest structural-model section in the document (id lands in
   *  its ResultSection; a re-run adds a newer section, so take the last). */
  const onSeeComparison = () => {
    const els = document.querySelectorAll<HTMLElement>('#pk-model');
    els[els.length - 1]?.scrollIntoView({ block: 'start' });
  };
  const evidence = gateDef?.key === 'adversarial_review' && state?.review_results
    ? <ReviewCard r={state.review_results} />
    : gateDef?.key === 'fit_pk_model' && state?.pk_model_results
      ? <PkModelEvidence r={state.pk_model_results}
          seal={latestEntryFor(audit, 'fit_pk_model')?.entry_hash ?? null}
          onSeeComparison={onSeeComparison} />
      : null;
  const outcome: Outcome | null =
    wfStatus === 'complete' && state?.report_path && session ? {
      kind: 'report',
      filename: state.report_path.split('/').pop() ?? 'report.docx',
      onDocx: () => { downloadExistingReport(); },
      onCsv: state.nca_parameters?.length ? () => { exportCsv('nca'); } : undefined,
      onCdisc: state.nca_parameters?.length ? () => { exportCdisc(); } : undefined,
    }
    : wfStatus === 'complete' ? { kind: 'complete' }
    : decisionView && wfStatus === 'running' ? {
      kind: 'running',
      note: decisionView === 'approved'
        ? (longLeg
          ? 'Approved — running the population fit (NLME → SCM → diagnostics → forest → VPC)'
          : 'Approved — generating report')
        : '',
    }
    : decisionView === 'rejected' && wfStatus === 'idle' ? {
      kind: 'rejected',
      stepLabel: gateDef?.label ?? 'the gate',
      entryIndex: [...audit].reverse().find(e => e.tool === 'human_review' && e.action.startsWith('rejected'))?.index ?? null,
      hint: activeWorkflow === 'nca_full'
        ? 'Refit λz or replace the dataset, then run again.'
        : 'Adjust the analysis, then run again.',
    }
    : null;
  /** Jump to the latest λz section in the document (id lands in S4; a re-run
   *  adds a newer section, so take the last) and focus its first control. */
  const onReviewLz = () => {
    const els = document.querySelectorAll<HTMLElement>('#lz');
    const el = els[els.length - 1];
    el?.scrollIntoView({ block: 'start' });
    (el?.querySelector('button') as HTMLElement | null)?.focus();
  };
  /** Hand focus to the review panel's heading. ≤1180px the panel is a drawer
   *  (visibility:hidden when closed), so open it first; ReviewPanel then
   *  focuses the heading itself when `open` flips. */
  const onReviewInPanel = () => {
    if (window.matchMedia(REVIEW_DRAWER_MQ).matches) setReviewOpen(true);
    document.getElementById('review-heading')?.focus();
  };
  /** Closing the drawer hands focus back to the topbar toggle (only rendered ≤1180px). */
  const closeReview = () => {
    setReviewOpen(false);
    document.getElementById('review-toggle')?.focus();
  };

  // ── Document (centre column) inputs ──
  /** Live seal for a marker's tool: the latest audit entry sealed under that
   *  backend tool name (refreshed by the state-keyed audit effect). Never a
   *  client-side hash; null when the marker has no registered tool. Only for
   *  cards that render live state — a snapshot card shows the seal bound to
   *  its own snapshot (bindSeals), so a re-run never relabels older numbers. */
  const sealFor = (k: Marker) => latestEntryFor(audit, CARD_META[k].tool)?.entry_hash ?? null;
  /** ResultSection attribution props for one marker message. */
  const sectionProps = (k: Marker, m: DisplayMsg) => {
    const meta = CARD_META[k];
    const seal = m.snap ? (m.seal ?? null) : sealFor(k);
    return {
      id: meta.anchor, title: meta.title, agent: m.agent ?? meta.agent,
      tool: meta.tool, seal, sealing: !!meta.tool && !seal,
    };
  };
  const num = (v: unknown): number | null => (typeof v === 'number' && Number.isFinite(v) ? v : null);
  const nSubjects = num(datasetMeta?.n_subjects);
  const docTitle = (messages.length === 0 && !file) ? 'New analysis'
    : ({ nca_full: 'NCA analysis', poppk_modeling: 'Modeling analysis', poppk_full: 'Population PK analysis' })[activeWorkflow];
  const startedAt = session?.created_at ? hhmm(session.created_at) : '';
  const docSubtitle = [
    file?.name,
    nSubjects != null && `${nSubjects} subjects`,
    startedAt && `started ${startedAt}`,
    audit.length > 0 && `${audit.length} sealed entries`,
  ].filter(Boolean).join(' · ') || 'Upload a PK dataset and choose a workflow.';
  /** Dataset profile stats — only values that exist in PharmState (no dashes; units only when the dataset labels them). */
  const profileItems: { label: string; value: string }[] = [];
  if (state?.dataset_metadata) {
    const dq = (state.data_quality ?? {}) as Record<string, unknown>;
    const nRecords = num(datasetMeta?.n_records);
    const nObs = num(dq.n_observations);
    const blq = num(state.spaghetti_data?.blq_excluded) ?? num(state.nca_summary?.blq?.n_below_loq);
    const doses = (state.nca_parameters ?? []).map(r => r.dose).filter((d): d is number => num(d) != null);
    if (nSubjects != null) profileItems.push({ label: 'Subjects', value: String(nSubjects) });
    if (nRecords != null) profileItems.push({ label: 'Records', value: String(nRecords) });
    if (nObs != null) profileItems.push({ label: 'Observations', value: String(nObs) });
    if (blq != null) profileItems.push({ label: 'Below LLOQ, excluded', value: String(blq) });
    if (doses.length) {
      const lo = sig3(Math.min(...doses)), hi = sig3(Math.max(...doses));
      const doseUnit = ncaUnits(state).dose;
      profileItems.push({ label: 'Dose range', value: `${lo === hi ? lo : `${lo}–${hi}`}${doseUnit ? ` ${doseUnit}` : ''}` });
    }
  }
  const profileSeal = latestEntryFor(audit, 'profile_pk_dataset')?.entry_hash ?? null;
  /** Export menu items — the same handlers and conditions as the former export chips. */
  const exportItems = [
    { id: 'docx', label: 'Full report (DOCX)', onClick: () => { downloadFullReport(); } },
    ...exportKinds.map(k => ({ id: `csv-${k}`, label: `${EXPORT_LABEL[k] ?? k} CSV`, onClick: () => { exportCsv(k); } })),
    ...(state?.nca_parameters?.length
      ? [{ id: 'cdisc', label: 'CDISC ADaM (zip)', onClick: () => { exportCdisc(); } }] : []),
    ...(state?.nlme_results?.status === 'ok' ? [
      { id: 'nonmem', label: 'NONMEM (.ctl)', title: 'NONMEM control stream (.ctl) seeded from the population fit',
        onClick: () => { exportControl('nonmem'); } },
      { id: 'mrgsolve', label: 'mrgsolve (.cpp)', title: 'mrgsolve model (.cpp) seeded from the population fit',
        onClick: () => { exportControl('mrgsolve'); } },
    ] : []),
  ];

  return (
    <>
      <Topbar
        sessionId={session?.id ?? null}
        title={file?.name ?? 'New session'}
        healthy={healthy}
        statusLabel={statusLabel}
        llmOpen={llmOpen}
        onLlmToggle={() => setLlmOpen(o => !o)}
        onLlmClose={() => setLlmOpen(false)}
        llmPanel={<LlmSettings onApplied={l => setLlmLabel(l)} onClose={() => setLlmOpen(false)} />}
        token={token}
        onTokenChange={v => { setTokenState(v); setToken(v); }}
        review={{
          open: reviewOpen,
          pending: decisionView === 'pending' && wfStatus === 'awaiting_review',
          onToggle: () => setReviewOpen(o => !o),
        }}
      />

      <Rail
        workflowName={activeWorkflow}
        steps={wfSteps}
        currentStep={currentStep}
        gateStep={gateStep}
        wfStatus={wfStatus}
        decision={decisionView}
        timings={timings}
        file={file}
        datasetMeta={datasetMeta}
        fileInput={<input ref={fileRef} type="file" accept=".csv" hidden onChange={onFileChange} />}
        dropZone={
          <button
            type="button"
            className={`upload-zone ${drag ? 'drag' : ''}`}
            aria-labelledby="upload-zone-label"
            aria-describedby="upload-zone-hint"
            onDragOver={e => { e.preventDefault(); setDrag(true); }}
            onDragLeave={() => setDrag(false)}
            onDrop={onDrop}
            onClick={() => fileRef.current?.click()}
          >
            <Upload size={22} style={{ color: 'var(--text-dim)', marginBottom: 8 }} aria-hidden="true" />
            <span id="upload-zone-label" className="upload-label">
              <strong>Click to upload</strong> or drag & drop
            </span>
            <span id="upload-zone-hint" className="upload-hint">CSV · NONMEM-style (ID / TIME / DV / AMT columns)</span>
          </button>
        }
        canReplace={wfStatus === 'idle'}
        onReplace={() => fileRef.current?.click()}
        onToggleRoles={() => setShowRoles(s => !s)}
        rolesEnabled={hasData}
        workflows={[
          { key: 'nca_full', label: 'Run NCA workflow', primary: true },
          { key: 'poppk_modeling', label: 'Run modeling + engines',
            title: 'Fit structural models, then confirm across estimation engines (native FOCE-I + nlmixr2 + Monolix when installed), reviewed and QC-gated' },
          { key: 'poppk_full', label: 'Run full population PK',
            title: 'Full population PK: structural comparison (gated), NLME fit, SCM covariate build, residual diagnostics, covariate forest, VPC, adversarial review (gated), report' },
        ]}
        canRun={canRunWorkflow}
        running={loading && wfStatus === 'running' ? activeWorkflow : null}
        onRun={uploadAndRun}
        onNewSession={newSession}
      />

      <main className="main">
        {/* The human-review gate and the report card live in the review panel (S3). */}
        <div className="messages">
          <DocumentHeader
            title={docTitle}
            subtitle={docSubtitle}
            right={<ExportMenu disabled={loading || exportKinds.length === 0} items={exportItems} />}
          />
          {state?.dataset_metadata && (
            <ResultSection title="Dataset profile" agent="data_manager" tool="profile_pk_dataset"
              seal={profileSeal} sealing={!profileSeal}>
              <DatasetProfile items={profileItems} />
            </ResultSection>
          )}
          {messages.length === 0 && (
            <div className="empty">
              <FlaskConical size={36} />
              <span>Upload a PK dataset and choose a workflow.</span>
              <span style={{ fontSize: 12 }}>Or ask the agents directly.</span>
            </div>
          )}
          {messages.map(m => {
            // Read from the message's frozen snapshot when present (set at the
            // time the card was produced), else the live state.
            const st = m.snap ?? state;
            if (m.content === '__SPAGHETTI__' && st?.spaghetti_data) {
              return (
                <ResultSection key={m.id} {...sectionProps('__SPAGHETTI__', m)}>
                  <SpaghettiChart data={st.spaghetti_data as SpaghettiData} flagged={Object.keys(flaggedExtrap(st))} />
                </ResultSection>
              );
            }
            if (m.content === '__NCA_TABLE__' && st?.nca_summary) {
              return (
                <ResultSection key={m.id} {...sectionProps('__NCA_TABLE__', m)}>
                  <NcaTable state={st} />
                </ResultSection>
              );
            }
            if (m.content === '__NCA_LZ__' && st?.nca_plot_data) {
              return (
                <ResultSection key={m.id} {...sectionProps('__NCA_LZ__', m)}>
                  <NcaLzPlot data={st.nca_plot_data as NcaPlotData} sessionId={session?.id ?? ''}
                    flagged={flaggedExtrap(st)} hours={/^(h|hrs?|hours?)$/i.test(ncaUnits(st).time ?? '')} />
                </ResultSection>
              );
            }
            if (m.content === '__QC_CARD__' && st?.qc_verdict) {
              return (
                <ResultSection key={m.id} {...sectionProps('__QC_CARD__', m)}>
                  <QcCard state={st} onReviewLz={onReviewLz} onReviewInPanel={qc ? onReviewInPanel : undefined} />
                </ResultSection>
              );
            }
            if (m.content === '__BE__' && st?.be_results) {
              return (
                <ResultSection key={m.id} {...sectionProps('__BE__', m)}>
                  <BeCard r={st.be_results} />
                </ResultSection>
              );
            }
            if (m.content === '__DP__' && st?.dose_prop_results) {
              return (
                <ResultSection key={m.id} {...sectionProps('__DP__', m)}>
                  <DosePropCard r={st.dose_prop_results} />
                </ResultSection>
              );
            }
            if (m.content === '__CLINPHARM__' && st?.clinpharm_results) {
              return (
                <ResultSection key={m.id} {...sectionProps('__CLINPHARM__', m)} maxWidth={700}>
                  <ClinpharmCard r={st.clinpharm_results} />
                </ResultSection>
              );
            }
            if (m.content === '__STATS__' && st?.stats_advice) {
              return (
                <ResultSection key={m.id} {...sectionProps('__STATS__', m)}>
                  <StatsAdviceCard r={st.stats_advice} />
                </ResultSection>
              );
            }
            if (m.content === '__COMPARTMENTAL__' && st?.compartmental_results) {
              return (
                <ResultSection key={m.id} {...sectionProps('__COMPARTMENTAL__', m)}>
                  <CompartmentalCard r={st.compartmental_results} />
                </ResultSection>
              );
            }
            if (m.content === '__POPPK__' && st?.poppk_results) {
              return (
                <ResultSection key={m.id} {...sectionProps('__POPPK__', m)}>
                  <PopPkCard r={st.poppk_results} />
                </ResultSection>
              );
            }
            if (m.content === '__PENDING_TOOL__' && st?.pending_tool) {
              const p = st.pending_tool;
              // buttons only while THIS proposal is still the live one
              const live = !!state?.pending_tool
                && state.pending_tool.tool === p.tool
                && state.pending_tool.proposed_at === p.proposed_at;
              return (
                <ResultSection key={m.id} {...sectionProps('__PENDING_TOOL__', m)} maxWidth={640}>
                  <div className="gate-banner">
                    <AlertTriangle size={20} style={{ color: 'var(--yellow)', flexShrink: 0 }} />
                    <div className="gate-banner-text">
                      <div className="gate-banner-title">Run {p.tool}?</div>
                      <div className="gate-banner-sub">
                        Long-running fit (minutes to tens of minutes) — runs as a background job.
                        Nothing has been computed.
                      </div>
                      <pre style={{ fontSize: 11, margin: '6px 0 0', whiteSpace: 'pre-wrap' }}>
                        {JSON.stringify(p.args, null, 1)}
                      </pre>
                    </div>
                    {live ? (
                      <div className="gate-actions">
                        <button className="btn btn-green" disabled={loading} onClick={() => decidePendingTool(true)}>
                          <CheckCircle size={12} /> Approve
                        </button>
                        <button className="btn btn-red" disabled={loading} onClick={() => decidePendingTool(false)}>
                          <XCircle size={12} /> Reject
                        </button>
                      </div>
                    ) : (
                      <div className="gate-banner-sub">decided</div>
                    )}
                  </div>
                </ResultSection>
              );
            }
            if (m.content === '__PKMODEL__' && st?.pk_model_results) {
              return (
                <ResultSection key={m.id} {...sectionProps('__PKMODEL__', m)}>
                  <PkModelCard r={st.pk_model_results} />
                </ResultSection>
              );
            }
            if (m.content === '__VPC__' && st?.vpc_results) {
              const wide = !!(st.vpc_results.stratified || st.vpc_results.exposure_pc
                || st.vpc_results.blq_vpc);
              return (
                <ResultSection key={m.id} {...sectionProps('__VPC__', m)} maxWidth={wide ? 920 : 640}>
                  <VpcCard r={st.vpc_results} onRerun={runVpc} busy={loading}
                    covariates={vpcStrataOptions(st.dataset_metadata as
                      { columns?: { name: string }[]; detected_roles?: Record<string, string> } | null)} />
                </ResultSection>
              );
            }
            if (m.content === '__NLME__' && st?.nlme_results) {
              return (
                <ResultSection key={m.id} {...sectionProps('__NLME__', m)} maxWidth={640}>
                  <NlmeCard r={st.nlme_results} />
                </ResultSection>
              );
            }
            if (m.content === '__PRIORCHECK__' && st?.prior_check_results) {
              return (
                <ResultSection key={m.id} {...sectionProps('__PRIORCHECK__', m)} maxWidth={640}>
                  <PriorCheckCard r={st.prior_check_results} />
                </ResultSection>
              );
            }
            if (m.content === '__SCM__' && st?.scm_results) {
              return (
                <ResultSection key={m.id} {...sectionProps('__SCM__', m)} maxWidth={680}>
                  <ScmCard r={st.scm_results} />
                </ResultSection>
              );
            }
            if (m.content === '__ENGINES__' && st?.engine_comparison_results) {
              return (
                <ResultSection key={m.id} {...sectionProps('__ENGINES__', m)} maxWidth={700}>
                  <EngineComparisonCard r={st.engine_comparison_results} />
                </ResultSection>
              );
            }
            if (m.content === '__FORECAST__' && st?.forecast_results) {
              return (
                <ResultSection key={m.id} {...sectionProps('__FORECAST__', m)} maxWidth={560}>
                  <ForecastCard r={st.forecast_results} />
                </ResultSection>
              );
            }
            if (m.content === '__DIAG__' && st?.diagnostics_results) {
              return (
                <ResultSection key={m.id} {...sectionProps('__DIAG__', m)} maxWidth={660}>
                  <DiagnosticsCard r={st.diagnostics_results} />
                </ResultSection>
              );
            }
            if (m.content === '__FOREST__' && st?.forest_results) {
              return (
                <ResultSection key={m.id} {...sectionProps('__FOREST__', m)} maxWidth={660}>
                  <ForestCard r={st.forest_results} />
                </ResultSection>
              );
            }
            if (m.content === '__SIMEST__' && st?.simest_results) {
              return (
                <ResultSection key={m.id} {...sectionProps('__SIMEST__', m)} maxWidth={660}>
                  <SimestCard r={st.simest_results} />
                </ResultSection>
              );
            }
            if (m.content === '__BOOTSTRAP__' && st?.bootstrap_results) {
              return (
                <ResultSection key={m.id} {...sectionProps('__BOOTSTRAP__', m)} maxWidth={760}>
                  <BootstrapCard r={st.bootstrap_results} />
                </ResultSection>
              );
            }
            if (m.content === '__SIR__' && st?.sir_results) {
              return (
                <ResultSection key={m.id} {...sectionProps('__SIR__', m)} maxWidth={700}>
                  <SirCard r={st.sir_results} />
                </ResultSection>
              );
            }
            if (m.content === '__PROFILE__' && st?.profile_results) {
              return (
                <ResultSection key={m.id} {...sectionProps('__PROFILE__', m)} maxWidth={760}>
                  <ProfileCard r={st.profile_results} />
                </ResultSection>
              );
            }
            if (m.content === '__SWEEP__' && st?.dose_sweep_results) {
              return (
                <ResultSection key={m.id} {...sectionProps('__SWEEP__', m)} maxWidth={640}>
                  <DoseSweepCard r={st.dose_sweep_results} />
                </ResultSection>
              );
            }
            if (m.content === '__CLINSIM__' && st?.clinsim_results) {
              return (
                <ResultSection key={m.id} {...sectionProps('__CLINSIM__', m)} maxWidth={660}>
                  <ClinsimCard r={st.clinsim_results} onRerun={runClinsim} busy={loading} />
                </ResultSection>
              );
            }
            if (m.content === '__EXPFOREST__' && st?.exposure_forest_results) {
              return (
                <ResultSection key={m.id} {...sectionProps('__EXPFOREST__', m)} maxWidth={680}>
                  <ExposureForestCard r={st.exposure_forest_results} />
                </ResultSection>
              );
            }
            if (m.content === '__SPECIALPOP__' && st?.special_pop_results) {
              return (
                <ResultSection key={m.id} {...sectionProps('__SPECIALPOP__', m)} maxWidth={720}>
                  <SpecialPopCard r={st.special_pop_results} onRerun={runSpecialPop} busy={loading} />
                </ResultSection>
              );
            }
            if (m.content === '__INDIVEXP__' && st?.individual_exposures) {
              return (
                <ResultSection key={m.id} {...sectionProps('__INDIVEXP__', m)} maxWidth={640}>
                  <IndividualExposuresCard r={st.individual_exposures} />
                </ResultSection>
              );
            }
            if (m.content === '__PEDIATRIC__' && st?.pediatric_results) {
              return (
                <ResultSection key={m.id} {...sectionProps('__PEDIATRIC__', m)} maxWidth={760}>
                  <PediatricCard r={st.pediatric_results} onRerun={runPediatric} busy={loading} />
                </ResultSection>
              );
            }
            if (m.content === '__SIM__' && st?.simulation_results) {
              return (
                <ResultSection key={m.id} {...sectionProps('__SIM__', m)} maxWidth={640}>
                  <SimChart sim={st.simulation_results} />
                </ResultSection>
              );
            }
            if (m.content === '__REVIEW__' && st?.review_results) {
              return (
                <ResultSection key={m.id} {...sectionProps('__REVIEW__', m)} maxWidth={720}>
                  <ReviewCard r={st.review_results} />
                </ResultSection>
              );
            }
            if (m.content === '__REPORT__') return null;
            return <MessageBubble key={m.id} msg={m} agent={m.agent} />;
          })}
          {/* Executed steps that put no section above (load, validate, review…), folded. */}
          {runLog.length > 0 && (
            <section className="section run-log">
              <details>
                <summary>
                  <ChevronRight size={14} className="run-log-chev" aria-hidden="true" />
                  <h2>Run log</h2>
                  <span className="section-attr">
                    {runLog.length} step{runLog.length === 1 ? '' : 's'} without a result section
                  </span>
                </summary>
                <ol className="run-log-list">
                  {runLog.map((s, i) => (
                    <li key={`${s.step}-${i}`}>
                      <div className="run-log-head">
                        <span className="run-log-label">{s.label}</span>
                        <span className="section-attr">{agentLabel(s.agent)} · <code>{s.tool}</code></span>
                      </div>
                      <div className="run-log-summary">{stripLabelPrefix(s.label, s.summary)}</div>
                    </li>
                  ))}
                </ol>
              </details>
            </section>
          )}
          {/* Panels toggled from the Actions menu render as document sections (no attribution). */}
          {showRoles && state && (
            <ResultSection title="Column roles">
              <RolesEditor state={state} onApply={applyRoles} loading={loading} />
            </ResultSection>
          )}
          {showFlexplot && session && (
            <ResultSection title="Flexplot">
              <FlexplotPanel sessionId={session.id} />
            </ResultSection>
          )}
          {showSkills && (
            <ResultSection title="Skills">
              <SkillsPanel skills={skills} loading={loading}
                datasetPath={state?.dataset_path ?? null}
                onRun={runSkill} onDelete={deleteSkill} onMarkdown={n => api.skillMarkdown(n)} />
            </ResultSection>
          )}
          {calcOpen && session && (
            <ResultSection title="Clinical pharmacology calculators">
              <CalculatorsPanel sessionId={session.id} busy={loading} onResult={(st, summary) => {
                setState(st);
                pushMsg({ role: 'assistant', content: summary, agent: 'clinpharm', id: '' });
                pushMsg({ role: 'assistant', content: '__CLINPHARM__', agent: 'clinpharm', id: '', snap: st });
              }} />
            </ResultSection>
          )}
          <div ref={messagesEnd} />
        </div>

        <Composer
          value={input}
          onChange={setInput}
          onKeyDown={onKeyDown}
          onSend={() => sendChat()}
          textDisabled={!session || loading}
          sendDisabled={!input.trim() || !session || loading}
          loading={loading}
          jobNote={jobNote}
          actionsOpen={actionsOpen}
          onToggleActions={() => setActionsOpen(o => !o)}
          actionsRef={actionsRef}
        >
          <ActionsMenu open={actionsOpen} onClose={() => setActionsOpen(false)} anchorRef={actionsRef}>
            {hasData && (
              <div className="quick-actions">
                <span className="quick-actions-label">Run on this data</span>
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

            {hasData && (
              <div className="quick-actions">
                <span className="quick-actions-label">Data</span>
                <button className="chip" data-keep-open disabled={loading} onClick={() => setShowRoles(s => !s)}>
                  {showRoles ? 'Hide columns' : 'Columns / roles'}
                </button>
              </div>
            )}

            {hasData && (
              <div className="quick-actions">
                <span className="quick-actions-label">Visualize</span>
                <button className="chip" data-keep-open disabled={loading} onClick={() => setShowFlexplot(s => !s)}>
                  {showFlexplot ? 'Hide flexplot' : 'Flexplot'}
                </button>
              </div>
            )}

            {hasData && pkModels.length > 0 && (
              <div className="quick-actions">
                <span className="quick-actions-label">PK model library</span>
                <select
                  className="model-select"
                  aria-label="PK model"
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
                <span className="quick-actions-label">Forecast</span>
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
                <span className="quick-actions-label">Population (NLME)</span>
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
                <button className="chip" disabled={loading || !state?.nlme_results}
                  onClick={() => runNlme('focei', { prior_from: 'nlme' })}
                  title="MAP fit with the current fit as an informative prior (Bayesian borrowing): informative prior from the stored fit's covariance. Fit adults first, then load the sparse (e.g. pediatric) data and click this.">
                  MAP (informative prior)
                </button>
                <button className="chip" disabled={loading || !state?.nlme_results}
                  onClick={() => runNlme('focei', { prior_from: 'nlme', prior_var: 1.0 })}
                  title="MAP fit with a WEAKLY informative prior (variance 1.0 on all params) — the estimate follows the data more than the prior.">
                  MAP (weak prior)
                </button>
                <button className="chip" disabled={loading || !(state?.nlme_results && state.nlme_results.map)}
                  onClick={runPriorCheck}
                  title="Prior-predictive band vs the data + prior-vs-posterior shrinkage. Needs a MAP fit.">
                  Prior check
                </button>
                <button className="chip" disabled={loading} onClick={runScm}
                  title="Stepwise covariate modeling: forward selection (p<0.05) + backward elimination (p<0.01) over dataset covariates">
                  Covariate SCM
                </button>
                <button className="chip" disabled={loading} onClick={runEngineComparison}
                  title="Fit the model across estimation engines (native FOCE-I + nlmixr2 + Monolix if installed); winner chosen by prediction accuracy, not cross-engine OFV">
                  Compare engines
                </button>
              </div>
            )}

            {state?.nlme_results?.status === 'ok' && (
              <div className="quick-actions">
                <span className="quick-actions-label">TDM / MAP forecast</span>
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
                  disabled={loading} aria-label="Forecast target metric" onChange={e => setFcMetric(e.target.value)}>
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
                <span className="quick-actions-label">Trial-design precision check</span>
                {!seShowConfirm ? (
                  <button className="chip" data-keep-open disabled={loading} onClick={() => setSeShowConfirm(true)}
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
                    <button className="chip" data-keep-open disabled={loading} onClick={() => setSeShowConfirm(false)}>
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
                <span className="quick-actions-label">Diagnostics</span>
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
                <button className="chip" disabled={loading} onClick={() => runSpecialPop()}
                  title="Special-population simulation: steady-state exposure by renal function (or covariate) vs the normal reference band → dose adjustment">
                  Special populations
                </button>
                <button className="chip" disabled={loading} onClick={runIndividualExposures}
                  title="Per-subject steady-state AUCss/Cmax,ss from the fitted EBEs (needs an NLME fit)">
                  Individual exposures
                </button>
                <button className="chip" disabled={loading} onClick={() => runPediatric()}
                  title="Pediatric dose-finding: age×weight exposure vs the adult range → the dose matching adult exposure (supports estimated allometry)">
                  Pediatric doses
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
                <span className="quick-actions-note">Dose sweep uses the Forecast q / # doses / to values.</span>
              </div>
            )}

            {(state?.nca_parameters?.length || state?.nlme_results?.status === 'ok') && (
              <div className="quick-actions">
                <span className="quick-actions-label">Review &amp; skills</span>
                <button className="chip" disabled={loading} onClick={runReview}
                  title="Adversarial reviewer: independently recompute and challenge every result; loop to a checkable goal">
                  Adversarial review
                </button>
                <button className="chip" disabled={loading} onClick={captureSkill}
                  title="Capture this session's analysis sequence as a reusable, replayable skill">
                  Capture as skill
                </button>
                <button className="chip" data-keep-open disabled={loading}
                  onClick={() => { setShowSkills(s => !s); if (!showSkills) refreshSkills(); }}>
                  {showSkills ? 'Hide skills' : `Skills${skills.length ? ` (${skills.length})` : ''}`}
                </button>
              </div>
            )}

            <div className="quick-actions">
              <span className="quick-actions-label">Calculators</span>
              <button className="chip" data-keep-open onClick={() => setCalcOpen(o => !o)}>{calcOpen ? 'Hide calculators' : 'Clinical pharmacology calculators'}</button>
            </div>
          </ActionsMenu>
        </Composer>
      </main>

      <ReviewPanel
        open={reviewOpen}
        onClose={closeReview}
        wfStatus={wfStatus}
        decision={decisionView}
        deciding={deciding}
        loading={loading}
        jobNote={jobNote}
        gate={gate}
        signer={gateSigner}
        onApprove={r => resume(true, r)}
        onReject={r => resume(false, r)}
        qc={qc}
        evidence={evidence}
        outcome={outcome}
        audit={audit}
        integrity={auditIntegrity}
        onVerify={verifyChain}
        verifyNote={verifyNote}
        onReviewLz={onReviewLz}
      />
    </>
  );
}
