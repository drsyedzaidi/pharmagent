import { useEffect, useRef, useState } from 'react';
import type { ReactNode } from 'react';
import { CheckCircle, Download, Loader2, XCircle } from 'lucide-react';
import type { AuditEntry, AuditIntegrityStatus, NcaSubject, QcCheck, QcIssue, WorkflowStatus } from '../types';
import type { Decision, GateSigner, Outcome, VerifyNote } from './types';
import { AuditTrail } from './AuditTrail';
import { QcEvidence } from './QcEvidence';

type Gate = { title: string; subtitle: string; stepLabel: string; placeholder: string };
type Qc = {
  verdict: string; checklist: QcCheck[] | null; issues: QcIssue[] | null;
  hash: string | null; subjects: NcaSubject[] | null;
};

const SPIN = { animation: 'spin 0.6s linear infinite' } as const;

/** Attribution copy must match what the backend actually seals: the actor is
 *  a pseudonymous token hash (never a name) or `anonymous`, per `GateSigner`. */
const SIGNER_COPY: Record<GateSigner, string> = {
  token: ' Sealed under your API-token identity (token:<hash>), not your name.',
  open: ' Sealed as anonymous: auth is open.',
  anonymous: ' Sealed as anonymous: no API token set.',
};

/** Approve / Reject with an optional reason. Local state only: the reason
 *  travels with the click, and the card unmounts once the gate is decided. */
function DecisionCard({ gate, signer, loading, onApprove, onReject }: {
  gate: Gate; signer: GateSigner; loading: boolean;
  onApprove: (reason: string) => void; onReject: (reason: string) => void;
}) {
  const [reason, setReason] = useState('');
  return (
    <section className="decision-card" aria-label="Decision">
      <div className="decision-card-titles">
        <span className="decision-card-title">{gate.title}</span>
        <span className="decision-card-sub">
          {gate.subtitle}
          {SIGNER_COPY[signer]}
        </span>
      </div>
      <label htmlFor="review-reason" className="decision-card-label">Reason (optional)</label>
      <textarea
        id="review-reason"
        className="decision-card-reason"
        rows={2}
        placeholder={gate.placeholder}
        value={reason}
        onChange={e => setReason(e.target.value)}
        disabled={loading}
      />
      <div className="decision-card-actions">
        <button type="button" className="btn btn-primary decision-approve" disabled={loading} onClick={() => onApprove(reason)}>
          <CheckCircle size={14} aria-hidden="true" /> Approve
        </button>
        <button type="button" className="btn btn-red decision-reject" disabled={loading} onClick={() => onReject(reason)}>
          <XCircle size={14} aria-hidden="true" /> Reject
        </button>
      </div>
    </section>
  );
}

function OutcomeCard({ outcome, jobNote }: { outcome: Outcome; jobNote: string }) {
  if (outcome.kind === 'report') {
    return (
      <section className="outcome-card report" aria-label="Report">
        <div className="outcome-card-title">
          <CheckCircle size={16} aria-hidden="true" /> Report ready
        </div>
        <span className="outcome-card-file">{outcome.filename}</span>
        <button type="button" className="btn btn-primary outcome-download" onClick={outcome.onDocx}>
          <Download size={14} aria-hidden="true" /> Download DOCX
        </button>
        {(outcome.onCsv || outcome.onCdisc) && (
          <div className="outcome-card-links">
            {outcome.onCsv && <button type="button" className="link-btn" onClick={outcome.onCsv}>NCA CSV</button>}
            {outcome.onCdisc && <button type="button" className="link-btn" onClick={outcome.onCdisc}>CDISC ADaM</button>}
          </div>
        )}
      </section>
    );
  }
  if (outcome.kind === 'running') {
    return (
      <section className="outcome-card running" aria-label="Progress">
        <div className="outcome-card-title">
          <Loader2 size={14} style={SPIN} aria-hidden="true" /> {outcome.note || 'Running…'}
        </div>
        {jobNote && <span className="outcome-card-sub">{jobNote}</span>}
      </section>
    );
  }
  if (outcome.kind === 'rejected') {
    return (
      <section className="outcome-card rejected" aria-label="Outcome">
        <span className="outcome-card-title">Run stopped at {outcome.stepLabel}</span>
        <span className="outcome-card-sub">
          {outcome.entryIndex != null && <>Rejection sealed as audit entry #{outcome.entryIndex}. </>}
          {outcome.hint}
        </span>
      </section>
    );
  }
  return (
    <section className="outcome-card complete" aria-label="Outcome">
      <span className="outcome-card-title">Workflow complete</span>
      <span className="outcome-card-sub">Use Export ▾ in the document header.</span>
    </section>
  );
}

/** Third column: the human-review gate lives here (not in the transcript),
 *  with the evidence it needs, the decision and its outcome, and the audit
 *  trail. Below 1180px it is a drawer toggled from the topbar. */
export function ReviewPanel({
  open, onClose, wfStatus, decision, deciding, loading, jobNote,
  gate, signer, onApprove, onReject,
  qc, evidence, outcome,
  audit, integrity, onVerify, verifyNote, onReviewLz,
}: {
  open: boolean;
  onClose: () => void;
  wfStatus: WorkflowStatus;
  decision: Decision;
  /** A gate decision is in flight: the card stays (disabled, reason kept)
   *  until the backend has taken it; the pill reports progress meanwhile. */
  deciding: boolean;
  loading: boolean;
  jobNote: string;
  gate: Gate | null;
  signer: GateSigner;
  onApprove: (reason: string) => void;
  onReject: (reason: string) => void;
  qc: Qc | null;
  evidence: ReactNode;
  outcome: Outcome | null;
  audit: AuditEntry[];
  integrity: AuditIntegrityStatus | null;
  onVerify: () => void;
  verifyNote: VerifyNote | null;
  onReviewLz: () => void;
}) {
  const headingRef = useRef<HTMLHeadingElement>(null);
  // A gate is the one moment the panel demands attention: move focus to it.
  // ≤1180px App opens the drawer in the same commit; a closed drawer is
  // visibility:hidden, so focus can never land on an offscreen heading.
  useEffect(() => {
    if (wfStatus === 'awaiting_review') headingRef.current?.focus();
  }, [wfStatus]);
  // Drawer opened (toggle or gate): focus moves into it; onClose hands it back.
  useEffect(() => {
    if (open) headingRef.current?.focus();
  }, [open]);
  // Escape closes the open drawer wherever focus is, not only inside the aside.
  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') onClose(); };
    document.addEventListener('keydown', onKey);
    return () => document.removeEventListener('keydown', onKey);
  }, [open, onClose]);

  const atGate = wfStatus === 'awaiting_review';
  const pill = atGate && deciding ? { cls: 'running', text: 'Running…' }
    : atGate && decision === 'pending' ? { cls: 'pending', text: 'Awaiting your decision' }
    : decision && wfStatus === 'running' ? { cls: 'running', text: 'Running…' }
    : decision === 'approved' && wfStatus === 'complete' ? { cls: 'approved', text: 'Approved' }
    : decision === 'rejected' ? { cls: 'rejected', text: 'Rejected' }
    : null;

  return (
    <aside
      className={`review ${open ? 'open' : ''}`}
      id="review-panel"
      aria-label="Review"
    >
      <div className="review-head">
        <h2 ref={headingRef} id="review-heading" tabIndex={-1}>Review</h2>
        <span className={`pill ${pill?.cls ?? ''}`} aria-live="polite">
          {pill?.cls === 'running' && <Loader2 size={12} style={SPIN} aria-hidden="true" />}
          {pill?.text ?? ''}
        </span>
      </div>

      {!gate && !outcome && !qc && !evidence && (
        <p className="review-empty">Nothing to review yet — a gated step will pause here.</p>
      )}

      {qc
        ? (
          <QcEvidence
            verdict={qc.verdict} checklist={qc.checklist} issues={qc.issues}
            seal={qc.hash} subjects={qc.subjects} onReviewLz={onReviewLz}
          />
        )
        : evidence}

      {gate && atGate && (
        <DecisionCard gate={gate} signer={signer} loading={loading} onApprove={onApprove} onReject={onReject} />
      )}

      {outcome && <OutcomeCard outcome={outcome} jobNote={jobNote} />}

      {audit.length > 0 && (
        <AuditTrail entries={audit} integrity={integrity} onVerify={onVerify} verifyNote={verifyNote} />
      )}
    </aside>
  );
}
