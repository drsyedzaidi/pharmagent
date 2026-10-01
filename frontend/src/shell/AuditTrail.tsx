import { useState } from 'react';
import { AlertTriangle, CheckCircle, ShieldCheck, XCircle } from 'lucide-react';
import type { AuditEntry, AuditIntegrityStatus } from '../types';
import type { VerifyNote } from './types';
import { integrityFailures, shortHash } from './format';
import { AGENT_LABEL } from './agents';

const COLLAPSE_ABOVE = 12;
const COLLAPSED_ROWS = 8;

/** Humanised agent name: known ids map to their display label, unknown ids
 *  drop the snake_case underscores (`data_manager` -> `data manager`). */
function agentName(id: string): string {
  return AGENT_LABEL[id] ?? id.replace(/_/g, ' ');
}

/** Header verdict. Only the backend's `verified` (chain + MAC + anchor) earns
 *  "verified"; an enforced-mode chain that fails MAC or anchor is a tamper /
 *  divergence signal and is named as such, never softened to "hash-only". */
function integrityStatus(integrity: AuditIntegrityStatus | null) {
  if (integrity === null) return { text: 'not yet checked', tone: 'dim' as const };
  if (integrity.verified) return { text: 'verified', tone: 'good' as const };
  if (integrity.mode === 'hash_only' && integrity.chain_ok) {
    return { text: 'hash-only, unauthenticated', tone: 'dim' as const };
  }
  const failed = integrityFailures(integrity);
  return {
    text: failed.length ? `verification failed: ${failed.join(', ')}` : 'verification failed',
    tone: 'bad' as const,
  };
}

/** The session's hash-chained audit trail, read straight from the backend.
 *  Every value shown (index, agent, tool, hash, actor, reason) is a field of
 *  a sealed entry — nothing is derived client-side. */
export function AuditTrail({ entries, integrity, onVerify, verifyNote }: {
  entries: AuditEntry[];
  integrity: AuditIntegrityStatus | null;
  onVerify: () => void;
  verifyNote: VerifyNote | null;
}) {
  const [showAll, setShowAll] = useState(false);
  const status = integrityStatus(integrity);
  const collapsed = entries.length > COLLAPSE_ABOVE && !showAll;
  const rows = collapsed ? entries.slice(-COLLAPSED_ROWS) : entries;

  return (
    <section className="audit-trail" aria-labelledby="audit-trail-h">
      <div className="audit-trail-head">
        <h3 id="audit-trail-h">Audit trail</h3>
        <span className={`audit-trail-status ${status.tone}`}>
          {entries.length} {entries.length === 1 ? 'entry' : 'entries'} ·{' '}
          {status.tone === 'bad' && <AlertTriangle size={11} aria-hidden="true" />}
          {status.tone === 'good' && <ShieldCheck size={11} aria-hidden="true" />}
          {status.text}
        </span>
      </div>
      <ol className="audit-trail-list">
        {rows.map(e => (
          <li key={e.index} className="audit-trail-row" title={e.reason ? `reason: ${e.reason}` : undefined}>
            <div className="audit-trail-main">
              <span className="audit-row-idx">#{e.index}</span>
              <span className="audit-row-agent">{agentName(e.agent)}</span>
              <span className="audit-row-tool">{e.tool}</span>
              <span className="audit-row-hash">{shortHash(e.entry_hash)}</span>
            </div>
            {((e.actor && e.actor !== 'anonymous') || e.reason) && (
              <div className="audit-trail-sub">
                {e.actor && e.actor !== 'anonymous' && <span className="audit-row-actor">by {e.actor}</span>}
                {e.reason && <span className="audit-row-reason">{e.reason}</span>}
              </div>
            )}
          </li>
        ))}
      </ol>
      <div className="audit-trail-foot">
        {collapsed && (
          <button type="button" className="link-btn" onClick={() => setShowAll(true)}>
            Show all {entries.length}
          </button>
        )}
        <button type="button" className="link-btn" onClick={onVerify}>Verify chain</button>
        {verifyNote && (
          <span className={`audit-verify-note ${verifyNote.tone}`} role="status">
            {verifyNote.tone === 'ok' && <CheckCircle size={12} aria-hidden="true" />}
            {verifyNote.tone === 'bad' && <XCircle size={12} aria-hidden="true" />}
            {' '}{verifyNote.text}
          </span>
        )}
      </div>
    </section>
  );
}
