import { useEffect, useRef, useState } from 'react';
import type { MouseEvent } from 'react';
import { hhmm } from '../shell/format';
import { evidenceDecideRecords, evidenceVerify } from './api';
import { FlagChip, StatusPill, VerifyBadge } from './badges';
import { BulkBar } from './BulkBar';
import { approval, bulkApproval, factsFromView, gateFor, ticked, type Gate } from './compareGate';
import { DecideRow } from './DecideRow';
import { clipValue, recordLabels, shaPrefix } from './format';
import { RecordContext } from './RecordContext';
import type {
  EvidenceSource, OnState, OnTool, RecordDecision, RecordRow, Run, TemplateConfig,
} from './types';
import { usePageImage } from './usePageImage';

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

/** A record's value: short values inline; a long one clipped, with the full text behind a disclosure
 *  that wraps inside the cell (a long value never widens the table past the decision controls). */
function RecordValue({ value, unit, name }: { value: string; unit: string | null; name: string }) {
  const clip = clipValue(value);
  return (
    <div className="ev-value-cell">
      <div className="ev-value-line">
        <span className="mono ev-value" title={clip.clipped ? value : undefined}>{clip.text}</span>
        {unit && <span className="ev-dim"> {unit}</span>}
      </div>
      {clip.clipped && (
        <details className="ev-details">
          <summary>Full value<span className="sr-only"> of {name}</span></summary>
          <p className="mono ev-value-long">{value}</p>
        </details>
      )}
    </div>
  );
}

/** Records tab: proposed records with their literal-match verification and QC flags, filters,
 *  per-record decisions with a required reason, bulk approval of verified unflagged rows. The
 *  cited page opens directly under its record, above that record's decision, so the evidence and
 *  the decision are read together. A record that failed verification cannot be approved (the
 *  server refuses it too); one that conflicts with an approved value needs an explicit keep-both. */
export function RecordsPanel({ sessionId, records, sources, template, disabled, run, onState, onTool, onLayoutChanged }: {
  sessionId: string;
  records: RecordRow[];
  sources: EvidenceSource[];
  template: TemplateConfig | null;
  disabled: boolean;
  run: Run;
  onState: OnState;
  onTool: OnTool;
  /** Called after a layout rebuild settles (refresh snapshot + state). */
  onLayoutChanged: () => void;
}) {
  const [filter, setFilter] = useState<Filter>('all');
  const [ctxId, setCtxId] = useState<string | null>(null);
  const [reasons, setReasons] = useState<Record<string, string>>({});
  const [keepBoth, setKeepBoth] = useState<Set<string>>(new Set());
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [bulkReason, setBulkReason] = useState('');
  const [status, setStatus] = useState('');
  // The compare ticks: record id -> the gate key it was given for. Never persisted; a key that no longer
  // matches the record's gate (another value, page or image) is deleted at once, not merely ignored.
  const [acks, setAcks] = useState<Record<string, string>>({});
  const [bulkAck, setBulkAck] = useState<string | null>(null);
  const deciding = useRef(false);                            // one decision call at a time, however it was triggered
  // The page image of the open record lives here, so its facts reach the gate in the same render.
  const view = usePageImage(sessionId, ctxId);
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
  // Names over every record (not only the filtered rows), so a row keeps its name across filters.
  const labels = recordLabels(records, names);
  const rowName = (r: RecordRow): string => labels.get(r.id) ?? r.id;
  const test = (FILTERS.find(f => f.key === filter) ?? FILTERS[0]).test;
  // Stable order (no status key): a decided record stays where it was.
  const rows = records.filter(test).sort((a, b) =>
    (names.get(a.source_id) ?? '').localeCompare(names.get(b.source_id) ?? '')
    || a.field.localeCompare(b.field) || a.group_label.localeCompare(b.group_label) || a.id.localeCompare(b.id));
  const gates = new Map<string, Gate>(records.map(r => [r.id, gateFor(r, r.id === ctxId ? factsFromView(view) : { state: 'closed' })]));
  const gateOf = (r: RecordRow): Gate => gates.get(r.id) ?? gateFor(r, { state: 'closed' });
  const isTicked = (r: RecordRow): boolean => ticked(gateOf(r), acks[r.id] ?? null);
  const approvalFor = (r: RecordRow) => approval(gateOf(r), isTicked(r), reasons[r.id] ?? '');
  const eligible = rows.filter(r => bulkEligible(r) && ['none', 'pdf'].includes(gateOf(r).mode));
  const picked = eligible.filter(r => selected.has(r.id));
  const bulkKey = picked.map(r => `${r.id}\u0000${gateOf(r).key ?? ''}`).join('\u0001');
  const bulkTicked = bulkAck !== null && bulkAck === bulkKey;
  const bulk = bulkApproval(picked.map(r => ({ id: r.id, gate: gateOf(r) })), bulkTicked, bulkReason);
  const proposed = rows.filter(r => r.status === 'proposed');
  // A tick is for exactly the record, value, page and image it was given for: drop every other one now.
  const staleAcks = Object.keys(acks).filter(id => acks[id] !== (gates.get(id)?.key ?? null));
  if (staleAcks.length > 0) setAcks(prev => Object.fromEntries(Object.entries(prev).filter(([id]) => !staleAcks.includes(id))));
  if (bulkAck !== null && bulkAck !== bulkKey) setBulkAck(null);

  const decide = async (decisions: RecordDecision[]) => {
    if (deciding.current) return false;                      // a second click never sends a second call
    deciding.current = true;
    try {
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
      setAcks(prev => Object.fromEntries(Object.entries(prev).filter(([k]) => !done.has(k))));
      setSelected(prev => new Set([...prev].filter(id => !done.has(id))));
      return true;
    } finally {
      deciding.current = false;
    }
  };
  /** Approve one record. The gate is checked again here, from the same state the controls were drawn from,
   *  so a blocked Approve sends nothing however it was activated. The reason sent is the composed one. */
  const approveOne = (r: RecordRow) => {
    const a = approvalFor(r);
    if (!a.allowed || r.verify_status === 'fail') return;
    void decide([{ record_id: r.id, decision: 'approve', reason: a.reason, ...(keepBoth.has(r.id) ? { allow_conflict: true } : {}) }]);
  };
  const rejectOne = (r: RecordRow) => {
    const reason = (reasons[r.id] ?? '').trim();
    if (reason) void decide([{ record_id: r.id, decision: 'reject', reason }]);
  };
  const approveBulk = () => {
    if (!bulk.allowed) return;
    void decide(picked.map(r => ({ record_id: r.id, decision: 'approve' as const, reason: bulk.reasons[r.id] })))
      .then(ok => { if (ok) setBulkReason(''); });
  };
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
            <button key={f.key} type="button" aria-pressed={filter === f.key} onClick={() => { setFilter(f.key); setAcks({}); setBulkAck(null); }}>
              {f.label} ({records.filter(f.test).length})
            </button>
          ))}
        </div>
        <button type="button" className="btn btn-ghost ev-push" disabled={disabled} onClick={() => void reverify()}
          title="Re-run the literal-match verifier on every proposed record">
          Re-verify proposed
        </button>
      </div>
      <BulkBar eligible={eligible.length} picked={picked.length} bulk={bulk} isTicked={bulkTicked} reason={bulkReason}
        disabled={disabled}
        onToggleAll={() => setSelected(picked.length === eligible.length ? new Set() : new Set(eligible.map(r => r.id)))}
        onReason={setBulkReason} onTick={on => setBulkAck(on ? bulkKey : null)} onApprove={approveBulk} />
      <p className="sr-only" role="status">{status}</p>

      <div className="ev-table-wrap">
        <table className="ev-table ev-records-table">
          <caption className="sr-only" ref={captionRef} tabIndex={-1}>Extracted records</caption>
          {/* Fixed layout: the table is never wider than its wrap, so Approve / Reject stay in view. */}
          <colgroup>
            <col className="ev-col-select" /><col className="ev-col-value" /><col /><col /><col className="ev-col-status" />
          </colgroup>
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
                    <RecordValue value={r.value} unit={r.unit} name={name} />
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
                        view={view} source={sources.find(x => x.id === r.source_id) ?? null}
                        autoFocus onClose={close} onLayoutChanged={onLayoutChanged} />
                    </td>
                  </tr>
                )}
                {r.status === 'proposed' && (
                  <tr className="ev-decide-row">
                    <td />
                    <td colSpan={COLS - 1}>
                      <DecideRow rec={r} name={name} gate={gateOf(r)} isTicked={isTicked(r)} approval={approvalFor(r)}
                        reason={reason} keepBoth={keepBoth.has(r.id)} hasConflict={hasConflict(r)} disabled={disabled}
                        reasonRef={keep(`reason:${r.id}`)} onReason={v => setReasons(prev => ({ ...prev, [r.id]: v }))}
                        onTick={on => setAcks(prev => {
                          const next = { ...prev };
                          const key = gateOf(r).key;
                          if (on && key !== null) next[r.id] = key; else delete next[r.id];
                          return next;
                        })}
                        onKeepBoth={() => toggleIn(setKeepBoth, r.id)} onApprove={() => approveOne(r)} onReject={() => rejectOne(r)} />
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
