import { useState } from 'react';
import type { LambdaZFlaggedSubject, LambdaZReview, LambdaZReviewDecision } from '../types';
import { buildDecisionBody, gateNeedsDecision, subjectStatus } from './lambdaZReview';

const STATUS_LABEL = { accepted: 'Accepted', revision: 'Revision requested', pending: 'Awaiting decision' } as const;

/** Human-only accept/revise gate for flagged λz fits. Shown from QC until every flagged subject is accepted;
 *  an agent cannot take this decision, so the buttons are the only way past it. */
export function LambdaZReviewGate({ review, flagged, busy, error, onDecide }: {
  review: LambdaZReview;
  flagged: LambdaZFlaggedSubject[];
  busy: boolean;
  error: string | null;
  onDecide: (body: ReturnType<typeof buildDecisionBody>) => void;
}) {
  const [picked, setPicked] = useState<Set<string>>(new Set());
  const [notes, setNotes] = useState('');
  const open = gateNeedsDecision(review, flagged);
  const toggle = (id: string) => setPicked(prev => {
    const next = new Set(prev);
    if (next.has(id)) next.delete(id); else next.add(id);
    return next;
  });
  const decide = (decision: LambdaZReviewDecision) => {
    const ids = flagged.filter(f => picked.has(String(f.id))).map(f => f.id);
    onDecide(buildDecisionBody(decision, ids, flagged, notes));
    setPicked(new Set());
    setNotes('');
  };
  const scope = picked.size ? `${picked.size} selected` : 'all flagged';

  return (
    <section className="decision-card" aria-label="λz review">
      <div className="decision-card-titles">
        <span className="decision-card-title">λz review{open ? ' — your decision' : ''}</span>
        <span className="decision-card-sub">
          {open
            ? 'Flagged terminal-slope fits block the report until you accept or revise them.'
            : review.status === 'accepted' ? 'All flagged fits accepted.' : 'No flagged fits.'}
        </span>
      </div>
      <ul className="lz-gate-list">
        {flagged.map(f => {
          const id = String(f.id);
          const st = subjectStatus(review, f.id);
          return (
            <li key={id}>
              <label>
                {open && <input type="checkbox" checked={picked.has(id)} disabled={busy} onChange={() => toggle(id)} />}
                {' '}Subject {id} <span className={`lz-gate-status lz-${st}`}>{STATUS_LABEL[st]}</span>
              </label>
              {f.reasons?.length ? <div className="decision-card-sub">{f.reasons.join('; ')}</div> : null}
            </li>
          );
        })}
      </ul>
      {open && (
        <>
          <label className="decision-card-label" htmlFor="lz-gate-notes">Notes (kept in the audit trail)</label>
          <textarea id="lz-gate-notes" className="decision-card-reason" rows={2} value={notes}
            disabled={busy} onChange={e => setNotes(e.target.value)} placeholder="e.g. sparse terminal sampling; fit reviewed" />
          {error && <div role="alert" className="decision-card-sub">{error}</div>}
          <div className="decision-card-actions">
            <button type="button" className="btn btn-primary decision-approve" disabled={busy}
              onClick={() => decide('accept')}>Accept ({scope})</button>
            <button type="button" className="btn btn-ghost" disabled={busy}
              onClick={() => decide('revise')}>Revise ({scope})</button>
          </div>
        </>
      )}
    </section>
  );
}
