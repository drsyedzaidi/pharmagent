import { useEffect, useRef, useState } from 'react';
import type { MouseEvent } from 'react';
import { hhmm } from '../shell/format';
import { evidenceDecideRecords, evidenceVerify } from './api';
import { FlagChip, StatusPill, VerifyBadge } from './badges';
import { failWords, recordLabel, shaPrefix } from './format';
import { RecordContext } from './RecordContext';
import type {
  Decision, EvidenceSource, OnState, OnTool, RecordDecision, RecordRow, Run, TemplateConfig,
} from './types';

const MAX_REASON = 500;
const COLS = 5;
type Filter = 'all' | 'flagged' | 'fail' | 'proposed';
const FILTERS: { key: Filter; label: string; test: (r: RecordRow) => boolean }[] = [
  { key: 'all', label: 'All', test: () => true },
  { key: 'flagged', label: 'Flagged', test: r => r.flags.length > 0 },
  { key: 'fail', label: 'Verify failed', test: r => r.verify_status === 'fail' },
  { key: 'proposed', label: 'Proposed', test: r => r.status === 'proposed' },
];

/** Bulk approval only ever covers verified, unflagged proposals; the rest need a look each. */
const bulkEligible = (r: RecordRow) => r.status === 'proposed' && r.verify_status === 'pass' && r.flags.length === 0;
const hasConflict = (r: RecordRow) => r.flags.some(f => f.code === 'duplicate');

/** Records tab: proposed records with their literal-match verification and QC flags, filters,
 *  per-record decisions with a required reason, bulk approval of verified unflagged rows. The
 *  cited page opens directly under its record, above that record's decision, so the evidence and
 *  the decision are read together. A record that failed verification cannot be approved (the
 *  server refuses it too); one that conflicts with an approved value needs an explicit keep-both. */
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
  const [keepBoth, setKeepBoth] = useState<Set<string>>(new Set());
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [bulkReason, setBulkReason] = useState('');
  const [status, setStatus] = useState('');
  const focusKey = useRef<string | null>(null);              // where focus goes once controls re-enable
  const focusables = useRef(new Map<string, HTMLElement>());
  const captionRef = useRef<HTMLTableCaptionElement>(null);
  const keep = (key: string) => (el: HTMLElement | null) => {
    if (el) focusables.current.set(key, el); else focusables.current.delete(key);
  };

  // Move focus once the target exists and is enabled (decisions disable every control meanwhile).
  useEffect(() => {
    const key = focusKey.current;
    if (!key || disabled) return;
    (key === 'caption' ? captionRef.current : focusables.current.get(key))?.focus();
    focusKey.current = null;
  });

  const names = new Map(sources.map(s => [s.id, s.filename]));
  const rowName = (r: RecordRow): string => recordLabel(r, names.get(r.source_id) ?? r.source_id);
  const test = (FILTERS.find(f => f.key === filter) ?? FILTERS[0]).test;
  // Stable order (no status key): a decided record stays where it was.
  const rows = records.filter(test).sort((a, b) =>
    (names.get(a.source_id) ?? '').localeCompare(names.get(b.source_id) ?? '')
    || a.field.localeCompare(b.field) || a.group_label.localeCompare(b.group_label) || a.id.localeCompare(b.id));
  const eligible = rows.filter(bulkEligible);
  const picked = eligible.filter(r => selected.has(r.id));
  const proposed = rows.filter(r => r.status === 'proposed');

  const decide = async (decisions: RecordDecision[]) => {
    const ok = await run(`Recording ${decisions.length} decision${decisions.length === 1 ? '' : 's'}…`, async () => {
      const res = await evidenceDecideRecords(sessionId, decisions);
      onState(res.state, res.summary);
    });
    if (!ok) return false;
    const done = new Set(decisions.map(d => d.record_id));
    const left = proposed.filter(r => !done.has(r.id));
    const after = proposed.findIndex(r => done.has(r.id));
    const next = left.find((_, i) => i >= after) ?? left[0];
    const first = rows.find(r => r.id === decisions[0].record_id);
    setStatus(`${decisions.length === 1 && first ? `${decisions[0].decision === 'approve' ? 'Approved' : 'Rejected'} ${rowName(first)}`
      : `${decisions.length} decisions recorded`}; ${left.length} proposed record${left.length === 1 ? '' : 's'} left.`);
    focusKey.current = next ? `reason:${next.id}` : 'caption';
    setReasons(prev => Object.fromEntries(Object.entries(prev).filter(([k]) => !done.has(k))));
    setSelected(prev => new Set([...prev].filter(id => !done.has(id))));
    return true;
  };
  const one = (r: RecordRow, decision: Decision) => void decide([{
    record_id: r.id, decision, reason: (reasons[r.id] ?? '').trim(),
    ...(decision === 'approve' && keepBoth.has(r.id) ? { allow_conflict: true } : {}),
  }]);
  const reverify = () => run('Re-verifying proposed records…', async () => onTool(await evidenceVerify(sessionId)));
  const toggleIn = (set: (f: (prev: Set<string>) => Set<string>) => void, id: string) => set(prev => {
    const next = new Set(prev);
    if (next.has(id)) next.delete(id); else next.add(id);
    return next;
  });
  const open = (id: string) => setCtxId(id);
  const close = () => { if (ctxId) focusKey.current = `link:${ctxId}`; setCtxId(null); };
  /** Row click opens the context, except on the row's own controls. */
  const onRow = (e: MouseEvent, id: string) => {
    if ((e.target as HTMLElement).closest('button, input, label, a, summary')) return;
    open(id);
  };

  if (records.length === 0) {
    return <p className="ev-empty">No records yet. Save a template and run extraction from the Template tab.</p>;
  }

  return (
    <div className="ev-panel">
      <p className="ev-intro">
        Every value was checked against the literal text of its cited page. Approve only what you have looked at;
        open a record to see its page, with the cited snippet marked, right under it.
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
      <p className="sr-only" role="status">{status}</p>

      <div className="ev-table-wrap">
        <table className="ev-table ev-records-table">
          <caption className="sr-only" ref={captionRef} tabIndex={-1}>Extracted records</caption>
          <thead>
            <tr>
              <th scope="col"><span className="sr-only">Select</span></th>
              <th scope="col">Field · value</th><th scope="col">Source</th><th scope="col">Checks</th>
              <th scope="col">Status</th>
            </tr>
          </thead>
          {rows.length === 0 && (
            <tbody><tr><td colSpan={COLS} className="ev-empty">No records match this filter.</td></tr></tbody>
          )}
          {rows.map(r => {
            const reason = reasons[r.id] ?? '';
            const failed = r.verify_status === 'fail';
            const name = rowName(r);
            const isOpen = r.id === ctxId;
            return (
              <tbody key={r.id} className={`ev-rowgroup${isOpen ? ' ev-row-current' : ''}`}>
                <tr onClick={e => onRow(e, r.id)}>
                  <td>
                    {bulkEligible(r) && (
                      <input type="checkbox" aria-label={`Select ${name}`} checked={selected.has(r.id)}
                        disabled={disabled} onChange={() => toggleIn(setSelected, r.id)} />
                    )}
                  </td>
                  <th scope="row" className="ev-row-head">
                    <button type="button" className="link-btn" ref={keep(`link:${r.id}`)} aria-expanded={isOpen}
                      aria-label={`${name}, show cited page`} title="Show the cited page under this record"
                      onClick={() => (isOpen ? close() : open(r.id))}>
                      {r.field}{r.group_label ? ` · ${r.group_label}` : ''}
                    </button>
                    <div><span className="mono ev-value">{r.value}</span>{r.unit && <span className="ev-dim"> {r.unit}</span>}</div>
                  </th>
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
                  <td>
                    <StatusPill status={r.status} />
                    {r.status !== 'proposed' && (
                      <div className="ev-prov">
                        {r.decided_by} · {hhmm(r.decided_at)}
                        <div className="ev-reason">“{r.decision_reason}”</div>
                      </div>
                    )}
                  </td>
                </tr>
                {isOpen && (
                  <tr className="ev-context-row">
                    <td colSpan={COLS}>
                      <RecordContext sessionId={sessionId} record={r} filename={names.get(r.source_id) ?? r.source_id}
                        autoFocus onClose={close} />
                    </td>
                  </tr>
                )}
                {r.status === 'proposed' && (
                  <tr className="ev-decide-row">
                    <td />
                    <td colSpan={COLS - 1}>
                      <div className="ev-decide">
                        <input className="ev-input" ref={keep(`reason:${r.id}`)} aria-label={`Reason for ${name}`}
                          value={reason} maxLength={MAX_REASON} placeholder="Reason (required)" disabled={disabled}
                          onChange={e => setReasons(prev => ({ ...prev, [r.id]: e.target.value }))} />
                        {hasConflict(r) && (
                          <label className="ev-check" title="An approved record already holds a different value for this field and group">
                            <input type="checkbox" checked={keepBoth.has(r.id)} disabled={disabled}
                              onChange={() => toggleIn(setKeepBoth, r.id)} /> Keep both values
                          </label>
                        )}
                        <button type="button" className="btn btn-primary" disabled={disabled || failed || !reason.trim()}
                          aria-label={`Approve ${name}`}
                          title={failed ? `Cannot approve: ${failWords(r)}. The value was not found on the cited page.`
                            : reason.trim() ? 'Approve this record' : 'A reason is required'}
                          onClick={() => one(r, 'approve')}>Approve</button>
                        <button type="button" className="btn btn-red" disabled={disabled || !reason.trim()}
                          aria-label={`Reject ${name}`}
                          title={reason.trim() ? 'Reject this record' : 'A reason is required'}
                          onClick={() => one(r, 'reject')}>Reject</button>
                      </div>
                    </td>
                  </tr>
                )}
              </tbody>
            );
          })}
        </table>
      </div>
    </div>
  );
}
