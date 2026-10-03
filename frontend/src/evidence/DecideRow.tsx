import { useId, useState } from 'react';
import { approval as approvalFor, LABEL_IMAGE, maxReasonLength, type Gate } from './compareGate';
import { failWords } from './format';
import type { RecordRow } from './types';

/** The decision controls of one proposed record: the reviewer's reason, the compare tick the gate asks for,
 *  Approve and Reject. Approve and Reject are `aria-disabled` rather than `disabled` while blocked: they stay
 *  reachable by keyboard and screen reader, and a hint (aria-describedby) says why. The panel re-checks the gate
 *  on every click, so a blocked button does nothing however it is activated.
 *
 *  The reason text is this row's own state (a keystroke re-renders one row, not the whole table); it is mirrored
 *  into `onReasonChange` so the row can move (under the page image when its record is open, back to the table
 *  when it closes) without losing what was typed. The tick stays mounted, disabled, while the page image loads,
 *  so focus never falls to the page and the change from blocked to ready is announced (the hint is a polite
 *  live region). */
export function DecideRow({ rec, name, gate, isTicked, reasonStart, keepBoth, hasConflict, disabled, reasonRef,
  onReasonChange, onTick, onKeepBoth, onApprove, onReject }: {
  rec: RecordRow;
  name: string;
  gate: Gate;
  isTicked: boolean;
  /** The reason typed before this row was (re)mounted, read once when it mounts. */
  reasonStart: () => string;
  keepBoth: boolean;
  hasConflict: boolean;
  disabled: boolean;
  reasonRef: (el: HTMLInputElement | null) => void;
  onReasonChange: (v: string) => void;
  onTick: (checked: boolean) => void;
  onKeepBoth: () => void;
  onApprove: (reason: string) => void;
  onReject: (reason: string) => void;
}) {
  const [reason, setReason] = useState(reasonStart);
  const hintId = useId();
  const rejectHintId = useId();
  const failed = rec.verify_status === 'fail';
  const pending = gate.mode === 'pending';
  const gated = gate.mode === 'image' || gate.mode === 'pdf' || pending;
  const label = gate.mode === 'image' || gate.mode === 'pdf' ? gate.label : LABEL_IMAGE;
  const approval = approvalFor(gate, isTicked, reason);
  const blocked = failed ? `Cannot approve: ${failWords(rec)}. The value was not found on the cited page.` : approval.blocked;
  const unavailable = blocked !== null;
  const noReason = !reason.trim();
  const change = (v: string) => { setReason(v); onReasonChange(v); };
  return (
    <>
      <div className="ev-decide">
        <input className="ev-input" ref={reasonRef} aria-label={`Reason for ${name}`} value={reason}
          maxLength={maxReasonLength(gate)} placeholder="Reason (required)" disabled={disabled}
          onChange={e => change(e.target.value)} />
        {hasConflict && (
          <label className="ev-check" title="An approved record already holds a different value for this field and group">
            <input type="checkbox" checked={keepBoth} disabled={disabled} onChange={onKeepBoth} /> Keep both values
          </label>
        )}
        <button type="button" className="btn btn-primary" disabled={disabled} aria-disabled={unavailable || undefined}
          aria-label={`Approve ${name}`} aria-describedby={hintId} onClick={() => { if (!unavailable) onApprove(reason); }}>
          Approve
        </button>
        <button type="button" className="btn btn-red" disabled={disabled} aria-disabled={noReason || undefined}
          aria-label={`Reject ${name}`} aria-describedby={rejectHintId} title={noReason ? 'A reason is required' : 'Reject this record'}
          onClick={() => { if (!noReason) onReject(reason); }}>
          Reject
        </button>
        <span id={rejectHintId} className="sr-only">{noReason ? 'A reason is required to reject.' : 'Rejects this record with your reason.'}</span>
      </div>
      <div className="ev-gate">
        {gated && (
          <label className="ev-check ev-gate-tick">
            <input type="checkbox" checked={isTicked} disabled={disabled || pending} aria-describedby={hintId}
              aria-label={`${label}, ${name}`}
              onChange={e => onTick(e.target.checked)} /> {label}
          </label>
        )}
        <p id={hintId} className="ev-gate-hint" role="status" aria-live="polite">{blocked ?? 'Ready to approve.'}</p>
        {approval.recorded !== null && gated && (
          <p className="ev-gate-hint ev-recorded">Will be recorded as: <q>{approval.recorded}</q></p>
        )}
      </div>
    </>
  );
}
