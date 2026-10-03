import { useId } from 'react';
import type { Approval, Gate } from './compareGate';
import { maxReasonLength } from './compareGate';
import { failWords } from './format';
import type { RecordRow } from './types';

/** The decision controls of one proposed record: the reviewer's reason, the compare tick the gate asks for,
 *  Approve and Reject. Approve is `aria-disabled` rather than `disabled` while blocked: it stays reachable by
 *  keyboard and screen reader, and the visible hint (aria-describedby) says why. The click handler of the
 *  panel re-checks the gate, so a blocked Approve does nothing however it is activated. */
export function DecideRow({ rec, name, gate, isTicked, approval, reason, keepBoth, hasConflict, disabled, reasonRef,
  onReason, onTick, onKeepBoth, onApprove, onReject }: {
  rec: RecordRow;
  name: string;
  gate: Gate;
  isTicked: boolean;
  approval: Approval;
  reason: string;
  keepBoth: boolean;
  hasConflict: boolean;
  disabled: boolean;
  reasonRef: (el: HTMLInputElement | null) => void;
  onReason: (v: string) => void;
  onTick: (checked: boolean) => void;
  onKeepBoth: () => void;
  onApprove: () => void;
  onReject: () => void;
}) {
  const hintId = useId();
  const failed = rec.verify_status === 'fail';
  const gated = gate.mode === 'image' || gate.mode === 'pdf';
  const blocked = failed ? `Cannot approve: ${failWords(rec)}. The value was not found on the cited page.` : approval.blocked;
  const unavailable = blocked !== null;
  return (
    <>
      <div className="ev-decide">
        <input className="ev-input" ref={reasonRef} aria-label={`Reason for ${name}`} value={reason}
          maxLength={maxReasonLength(gate)} placeholder="Reason (required)" disabled={disabled}
          onChange={e => onReason(e.target.value)} />
        {hasConflict && (
          <label className="ev-check" title="An approved record already holds a different value for this field and group">
            <input type="checkbox" checked={keepBoth} disabled={disabled} onChange={onKeepBoth} /> Keep both values
          </label>
        )}
        <button type="button" className="btn btn-primary" disabled={disabled} aria-disabled={unavailable || undefined}
          aria-label={`Approve ${name}`} aria-describedby={hintId} onClick={() => { if (!unavailable) onApprove(); }}>
          Approve
        </button>
        <button type="button" className="btn btn-red" disabled={disabled || !reason.trim()}
          aria-label={`Reject ${name}`} title={reason.trim() ? 'Reject this record' : 'A reason is required'} onClick={onReject}>
          Reject
        </button>
      </div>
      <div className="ev-gate">
        {gated && (
          <label className="ev-check ev-gate-tick">
            <input type="checkbox" checked={isTicked} disabled={disabled} aria-label={`${gate.label}, ${name}`}
              onChange={e => onTick(e.target.checked)} /> {gate.label}
          </label>
        )}
        <p id={hintId} className="ev-gate-hint">{blocked ?? 'Ready to approve.'}</p>
        {approval.recorded !== null && gated && (
          <p className="ev-gate-hint ev-recorded">Will be recorded as: <q>{approval.recorded}</q></p>
        )}
      </div>
    </>
  );
}
