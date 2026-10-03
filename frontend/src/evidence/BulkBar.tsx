import { useId } from 'react';
import type { BulkApproval } from './compareGate';
import { LABEL_PDF, MAX_REASON, PDF_PREFIX } from './compareGate';

/** Bulk approval of verified, unflagged records. Records that need their picture compared are not offered
 *  here (open and approve them one by one); the others share one tick for a comparison with the PDF itself,
 *  composed into each record's own reason. Approve is `aria-disabled` while blocked, with a visible hint. */
export function BulkBar({ eligible, leftOut, picked, bulk, isTicked, reason, disabled, onToggleAll, onReason, onTick, onApprove }: {
  eligible: number;
  /** Why verified, unflagged records are not offered (their PDF has no usable page layout); '' when none. */
  leftOut: string;
  picked: number;
  bulk: BulkApproval;
  isTicked: boolean;
  reason: string;
  disabled: boolean;
  onToggleAll: () => void;
  onReason: (v: string) => void;
  onTick: (checked: boolean) => void;
  onApprove: () => void;
}) {
  const hintId = useId();
  const all = picked === eligible && eligible > 0;
  const unavailable = !bulk.allowed;
  return (
    <>
      <div className="ev-actions">
        <button type="button" className="btn btn-ghost" disabled={disabled || eligible === 0} onClick={onToggleAll}>
          {all ? 'Clear selection' : `Select verified, unflagged (${eligible})`}
        </button>
        <input className="ev-input ev-bulk-reason" aria-label="Reason for the selected records" value={reason}
          maxLength={MAX_REASON - PDF_PREFIX.length} placeholder="Reason, applied to every selected record" disabled={disabled}
          onChange={e => onReason(e.target.value)} />
        <button type="button" className="btn btn-primary" disabled={disabled} aria-disabled={unavailable || undefined}
          aria-describedby={hintId} onClick={() => { if (!unavailable) onApprove(); }}>
          Approve selected ({picked})
        </button>
      </div>
      <div className="ev-gate">
        {bulk.needsTick && (
          <label className="ev-check ev-gate-tick">
            <input type="checkbox" checked={isTicked} disabled={disabled} onChange={e => onTick(e.target.checked)} />
            {' '}{LABEL_PDF.replace('the value', 'every selected value')}
          </label>
        )}
        <p id={hintId} className="ev-gate-hint">{bulk.blocked ?? 'Ready to approve the selected records.'}</p>
        {bulk.allowed && bulk.needsTick && (
          <p className="ev-gate-hint ev-recorded">
            Records covered by the tick are recorded as: <q>{PDF_PREFIX}{reason.trim()}</q>
          </p>
        )}
        {leftOut && <p className="ev-gate-hint">{leftOut}</p>}
      </div>
    </>
  );
}
