import { useEffect, useRef, useState } from 'react';
import type { MouseEvent } from 'react';
import { hhmm } from '../shell/format';
import { evidenceDecideRecords, evidenceVerify } from './api';
import { FlagChip, StatusPill, VerifyBadge } from './badges';
import { failWords, shaPrefix } from './format';
import { RecordContext } from './RecordContext';
import type {
  Decision, EvidenceSource, OnState, OnTool, RecordDecision, RecordRow, Run, TemplateConfig,
} from './types';

const MAX_REASON = 500;
type Filter = 'all' | 'flagged' | 'fail' | 'proposed';
const FILTERS: { key: Filter; label: string; test: (r: RecordRow) => boolean }[] = [
  { key: 'all', label: 'All', test: () => true },
  { key: 'flagged', label: 'Flagged', test: r => r.flags.length > 0 },
  { key: 'fail', label: 'Verify failed', test: r => r.verify_status === 'fail' },
  { key: 'proposed', label: 'Proposed', test: r => r.status === 'proposed' },
];
const STATUS_ORDER = { proposed: 0, approved: 1, rejected: 2 } as const;

/** Bulk approval only ever covers verified, unflagged proposals; the rest need a look each. */
const bulkEligible = (r: RecordRow) => r.status === 'proposed' && r.verify_status === 'pass' && r.flags.length === 0;

/** Records tab: proposed records with their literal-match verification and QC flags,
 *  filters, per-record decisions with a required reason, bulk approval of verified unflagged
 *  rows, and the page context of the selected record. A record that failed verification
 *  cannot be approved (the server refuses it too). */
export function RecordsPanel({ sessionId, records, sources, template, disabled, run, onState, onTool }: {
  sessionId: string;
  records: RecordRow[];
  sources: EvidenceSource[];
  template: TemplateConfig | null;
  disabled: boolean;
  run: Run;
  onState: OnState;
  onTool: OnTool;
}) {
  const [filter, setFilter] = useState<Filter>('all');
  const [ctxId, setCtxId] = useState<string | null>(null);
  const [reasons, setReasons] = useState<Record<string, string>>({});
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [bulkReason, setBulkReason] = useState('');
  const asideRef = useRef<HTMLElement>(null);
  // Stacked layout puts the context below the table: bring it into view (no animation).
  useEffect(() => { if (ctxId) asideRef.current?.scrollIntoView({ block: 'nearest' }); }, [ctxId]);

  const names = new Map(sources.map(s => [s.id, s.filename]));
  const test = (FILTERS.find(f => f.key === filter) ?? FILTERS[0]).test;
  const rows = records.filter(test).sort((a, b) =>
    STATUS_ORDER[a.status] - STATUS_ORDER[b.status]
    || (names.get(a.source_id) ?? '').localeCompare(names.get(b.source_id) ?? '')
    || a.field.localeCompare(b.field) || a.group_label.localeCompare(b.group_label));
  const eligible = rows.filter(bulkEligible);
  const picked = eligible.filter(r => selected.has(r.id));
  const ctx = records.find(r => r.id === ctxId) ?? null;

  const decide = async (decisions: RecordDecision[]) => {
    const ok = await run(`Recording ${decisions.length} decision${decisions.length === 1 ? '' : 's'}…`, async () => {
      const res = await evidenceDecideRecords(sessionId, decisions);
      onState(res.state, res.summary);
    });
    if (!ok) return false;
    const done = new Set(decisions.map(d => d.record_id));
    setReasons(prev => Object.fromEntries(Object.entries(prev).filter(([k]) => !done.has(k))));
    setSelected(prev => new Set([...prev].filter(id => !done.has(id))));
    return true;
  };
  const one = (id: string, decision: Decision) =>
    void decide([{ record_id: id, decision, reason: (reasons[id] ?? '').trim() }]);
  const reverify = () => run('Re-verifying proposed records…', async () => onTool(await evidenceVerify(sessionId)));
  const toggle = (id: string) => setSelected(prev => {
    const next = new Set(prev);
    if (next.has(id)) next.delete(id); else next.add(id);
    return next;
  });
  /** Row click opens the context, except on the row's own controls. */
  const onRow = (e: MouseEvent, id: string) => {
    if ((e.target as HTMLElement).closest('button, input, label, a, summary')) return;
    setCtxId(id);
  };

  if (records.length === 0) {
    return <p className="ev-empty">No records yet. Save a template and run extraction from the Template tab.</p>;
  }

  return (
    <div className="ev-panel">
      <p className="ev-intro">
        Every value was checked against the literal text of its cited page. Approve only what you have looked at;
        select a row to see its page with the cited snippet marked.
      </p>
      <div className="ev-actions">
        <div className="seg" role="group" aria-label="Filter records">
          {FILTERS.map(f => (
            <button key={f.key} type="button" aria-pressed={filter === f.key} onClick={() => setFilter(f.key)}>
              {f.label} ({records.filter(f.test).length})
            </button>
          ))}
        </div>
        <button type="button" className="btn btn-ghost ev-push" disabled={disabled} onClick={() => void reverify()}
          title="Re-run the literal-match verifier on every proposed record">
          Re-verify proposed
        </button>
      </div>
      <div className="ev-actions">
        <button type="button" className="btn btn-ghost" disabled={disabled || eligible.length === 0}
          onClick={() => setSelected(picked.length === eligible.length ? new Set() : new Set(eligible.map(r => r.id)))}>
          {picked.length === eligible.length && eligible.length > 0 ? 'Clear selection' : `Select verified, unflagged (${eligible.length})`}
        </button>
        <input className="ev-input ev-bulk-reason" aria-label="Reason for the selected records" value={bulkReason}
          maxLength={MAX_REASON} placeholder="Reason, applied to every selected record" disabled={disabled}
          onChange={e => setBulkReason(e.target.value)} />
        <button type="button" className="btn btn-primary" disabled={disabled || picked.length === 0 || !bulkReason.trim()}
          title={picked.length === 0 ? 'Select verified, unflagged records first' : !bulkReason.trim() ? 'A reason is required' : undefined}
          onClick={() => void decide(picked.map(r => ({ record_id: r.id, decision: 'approve', reason: bulkReason.trim() })))
            .then(ok => { if (ok) setBulkReason(''); })}>
          Approve selected ({picked.length})
        </button>
      </div>

      <div className="ev-records">
        <div className="ev-records-grid">
          <div className="ev-table-wrap">
            <table className="ev-table ev-records-table">
              <caption className="sr-only">Extracted records</caption>
              <thead>
                <tr>
                  <th scope="col"><span className="sr-only">Select</span></th>
                  <th scope="col">Field · value</th><th scope="col">Source</th><th scope="col">Checks</th>
                  <th scope="col">Status</th><th scope="col">Decision</th>
                </tr>
              </thead>
              <tbody>
                {rows.length === 0 && <tr><td colSpan={6} className="ev-empty">No records match this filter.</td></tr>}
                {rows.map(r => {
                  const reason = reasons[r.id] ?? '';
                  const failed = r.verify_status === 'fail';
                  return (
                    <tr key={r.id} className={r.id === ctxId ? 'ev-row-current' : undefined} onClick={e => onRow(e, r.id)}>
                      <td>
                        {bulkEligible(r) && (
                          <input type="checkbox" aria-label={`Select ${r.field} ${r.group_label}`} checked={selected.has(r.id)}
                            disabled={disabled} onChange={() => toggle(r.id)} />
                        )}
                      </td>
                      <td>
                        <button type="button" className="link-btn" aria-pressed={r.id === ctxId} onClick={() => setCtxId(r.id)}
                          title="Show the cited page">
                          {r.field}{r.group_label ? ` · ${r.group_label}` : ''}
                        </button>
                        <div><span className="mono ev-value">{r.value}</span>{r.unit && <span className="ev-dim"> {r.unit}</span>}</div>
                      </td>
                      <td>
                        <div className="ev-src-name">{names.get(r.source_id) ?? r.source_id}</div>
                        <div className="ev-prov">
                          p. {r.locator.page} · {r.locator.kind} · {r.extractor === 'llm' ? 'model' : r.extractor === 'table' ? 'rules' : 'manual'}
                          {' · '}<code>{shaPrefix(r.record_hash, 10)}</code>
                          {template && r.template_hash !== template.body_hash && ' · earlier template'}
                        </div>
                      </td>
                      <td>
                        <VerifyBadge rec={r} />
                        {r.flags.length > 0 && (
                          <div className="ev-flags">{r.flags.map((f, i) => <FlagChip key={`${f.code}-${i}`} flag={f} />)}</div>
                        )}
                      </td>
                      <td><StatusPill status={r.status} /></td>
                      <td className="ev-decide-cell">
                        {r.status === 'proposed' ? (
                          <div className="ev-decide">
                            <input className="ev-input" aria-label={`Reason for ${r.field} ${r.group_label}`} value={reason}
                              maxLength={MAX_REASON} placeholder="Reason (required)" disabled={disabled}
                              onChange={e => setReasons(prev => ({ ...prev, [r.id]: e.target.value }))} />
                            <button type="button" className="btn btn-primary" disabled={disabled || failed || !reason.trim()}
                              title={failed ? `Cannot approve: ${failWords(r)}. The value was not found on the cited page.`
                                : reason.trim() ? 'Approve this record' : 'A reason is required'}
                              onClick={() => one(r.id, 'approve')}>Approve</button>
                            <button type="button" className="btn btn-red" disabled={disabled || !reason.trim()}
                              title={reason.trim() ? 'Reject this record' : 'A reason is required'}
                              onClick={() => one(r.id, 'reject')}>Reject</button>
                          </div>
                        ) : (
                          <div className="ev-prov">
                            {r.status} by {r.decided_by} · {hhmm(r.decided_at)}
                            <div className="ev-reason">“{r.decision_reason}”</div>
                          </div>
                        )}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
          <aside ref={asideRef} className="ev-context-col" aria-label="Record context">
            {ctx
              ? <RecordContext key={ctx.id} sessionId={sessionId} record={ctx} filename={names.get(ctx.source_id) ?? ctx.source_id} />
              : <p className="ev-empty">Select a record to see its page with the cited snippet marked.</p>}
          </aside>
        </div>
      </div>
    </div>
  );
}
