import type { ReactNode } from 'react';
import { Activity, Check, CheckCircle, FileText, Loader2, X } from 'lucide-react';
import type { WorkflowStatus } from '../types';
import type { Decision, StepView, WorkflowName } from './types';

const WORKFLOW_DISPLAY: Record<string, string> = {
  nca_full: 'NCA',
  poppk_modeling: 'Modeling + engines',
  poppk_full: 'Population PK',
};

export type DatasetMeta = { n_records?: unknown; n_subjects?: unknown; n_columns?: unknown };

function metaLine(meta: DatasetMeta | null): string {
  if (!meta) return '';
  const parts: string[] = [];
  if (typeof meta.n_records === 'number') parts.push(`${meta.n_records} rows`);
  if (typeof meta.n_subjects === 'number') parts.push(`${meta.n_subjects} subjects`);
  if (typeof meta.n_columns === 'number') parts.push(`${meta.n_columns} columns`);
  return parts.join(' · ');
}

/** Left rail: workflow step tracker (with audit-derived timings), dataset card
 *  or drop zone, and the three workflow launch buttons. Presentational — the
 *  drop zone, hidden file input and every handler are supplied by App. */
export function Rail({
  workflowName, steps, currentStep, gateStep, wfStatus, decision, timings,
  file, datasetMeta, dropZone, fileInput, canReplace, onReplace, onToggleRoles, rolesEnabled,
  workflows, canRun, running, onRun, onNewSession,
}: {
  workflowName: string;
  steps: readonly StepView[];
  currentStep: number;
  gateStep: number;
  wfStatus: WorkflowStatus;
  decision: Decision;
  timings: Record<string, string>;
  file: File | null;
  datasetMeta: DatasetMeta | null;
  dropZone: ReactNode;
  fileInput: ReactNode;
  canReplace: boolean;
  onReplace: () => void;
  onToggleRoles: () => void;
  rolesEnabled: boolean;
  workflows: { key: WorkflowName; label: string; title?: string; primary?: boolean }[];
  canRun: boolean;
  running: WorkflowName | null;
  onRun: (key: WorkflowName) => void;
  onNewSession: () => void;
}) {
  const meta = metaLine(datasetMeta);
  return (
    <aside className="sidebar" aria-label="Workflow and dataset">
      <section className="sidebar-section" aria-labelledby="rail-workflow-h">
        <h2 id="rail-workflow-h" className="sidebar-label">Workflow</h2>
        <div className="rail-workflow-name">{WORKFLOW_DISPLAY[workflowName] ?? workflowName}</div>
        <ol className="step-list">
          {steps.map((s, i) => {
            const done = currentStep > i || wfStatus === 'complete';
            const active = currentStep === i && wfStatus === 'running';
            // DELIBERATE: only the step the run is paused at lights up. Previously
            // every `gate: true` step was tinted while awaiting review, so
            // poppk_full's second gate highlighted before it was reached.
            const gate = wfStatus === 'awaiting_review' && i === gateStep;
            const decided = (decision === 'approved' || decision === 'rejected')
              && i === gateStep && wfStatus !== 'awaiting_review';
            const rejected = decided && decision === 'rejected';
            const cls = gate ? 'gate' : done ? 'done' : active ? 'active' : 'pending';
            const status = gate ? 'needs your decision' : rejected ? 'rejected' : decided ? 'approved'
              : active ? 'running' : done ? 'done' : 'pending';
            // Gate and decided rows already carry a visible status word in
            // .step-trail; the others only have an aria-hidden icon, so they
            // get an sr-only word (same pattern as QcEvidence).
            const hasVisibleStatus = gate || decided;
            return (
              <li key={s.key} className={`step-item ${cls}`}
                aria-current={gate || active ? 'step' : undefined}
                title={`${s.label} — ${status}`}>
                <span className="step-icon" aria-hidden="true">
                  {gate ? <span className="step-dot" />
                    : rejected ? <X size={14} strokeWidth={2.4} style={{ color: 'var(--red)' }} />
                    : done ? <Check size={14} strokeWidth={2.4} style={{ color: 'var(--green)' }} />
                    : active ? <Loader2 size={14} style={{ animation: 'spin 1s linear infinite' }} />
                    : <span className="step-circle" />}
                </span>
                <span className="step-label">{s.label}</span>
                {!hasVisibleStatus && <span className="sr-only"> {status}</span>}
                {gate && <span className="step-trail needs-you">Needs you</span>}
                {decided && decision === 'approved' && (
                  <span className="step-trail approved"><Check size={11} strokeWidth={2.6} aria-hidden="true" /> Approved</span>
                )}
                {rejected && (
                  <span className="step-trail rejected"><X size={11} strokeWidth={2.6} aria-hidden="true" /> Rejected</span>
                )}
                {/* Timings only on completed rows: a session's audit keeps earlier
                    runs' seals, so a pending row could otherwise show a stale delta. */}
                {done && !active && !gate && !decided && timings[s.key] && (
                  <span className="step-timing" title="time between audit seals">{timings[s.key]}</span>
                )}
              </li>
            );
          })}
        </ol>
      </section>

      <section className="sidebar-section" aria-labelledby="rail-dataset-h">
        <h2 id="rail-dataset-h" className="sidebar-label">Dataset</h2>
        {fileInput}
        {!file ? dropZone : (
          <div className="dataset-card">
            <FileText size={16} style={{ color: 'var(--text-dim)', flexShrink: 0, marginTop: 1 }} aria-hidden="true" />
            <div className="dataset-body">
              <span className="dataset-name">
                <span className="dataset-filename">{file.name}</span>
                <CheckCircle size={12} style={{ color: 'var(--green)', flexShrink: 0 }} aria-label="selected" />
              </span>
              {meta && <span className="dataset-meta">{meta}</span>}
              <div className="dataset-links">
                <button type="button" className="link-btn" disabled={!rolesEnabled} onClick={onToggleRoles}
                  title={rolesEnabled ? undefined : 'Available once the dataset has been loaded'}>
                  Column roles
                </button>
                <button type="button" className="link-btn" disabled={!canReplace} onClick={onReplace}
                  title={canReplace ? undefined : 'Start a new session to change the dataset'}>
                  Replace
                </button>
              </div>
            </div>
          </div>
        )}
      </section>

      <div className="sidebar-section rail-actions">
        {workflows.map(w => (
          <button
            key={w.key}
            type="button"
            className={`workflow-btn ${w.primary ? 'primary' : ''}`}
            disabled={!canRun}
            title={w.title ?? w.label}
            onClick={() => onRun(w.key)}
          >
            {running === w.key
              ? <><div className="spinner" /> <span>Running…</span></>
              : <><Activity size={13} aria-hidden="true" /> <span>{w.label}</span></>}
          </button>
        ))}
        <button type="button" className="link-btn rail-new-session" onClick={onNewSession}
          title="Starts a fresh session and audit chain">
          New session
        </button>
      </div>
    </aside>
  );
}
