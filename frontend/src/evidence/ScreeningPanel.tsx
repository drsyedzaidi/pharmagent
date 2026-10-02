import { useEffect, useRef, useState } from 'react';
import { hhmm } from '../shell/format';
import { evidenceDecideScreen } from './api';
import { InclusionPill, MetWord, VerdictPill } from './badges';
import { inclusionOf, shaPrefix } from './format';
import type { CriteriaConfig, EvidenceSource, Inclusion, OnState, Run, ScreenDecision, ScreenRow } from './types';

const MAX_REASON = 500;

/** One verdict's reasoning and per-criterion evidence (page + snippet), folded. */
function Reasoning({ s, texts }: { s: ScreenRow; texts: Map<string, string> }) {
  return (
    <details className="ev-details">
      <summary>Reasoning and evidence</summary>
      <p className="ev-reasoning">{s.reasoning}</p>
      <ul className="ev-crit-list">
        {s.criteria.map(c => (
          <li key={c.criterion_id}>
            <div>
              <MetWord met={c.met} /> <code>{c.criterion_id}</code> {c.kind}{c.required ? ', required' : ''}
              {texts.get(c.criterion_id) && <> · {texts.get(c.criterion_id)}</>}
              {c.note && <span className="ev-dim"> · {c.note}</span>}
            </div>
            {(c.evidence ?? []).map((e, i) => (
              <div key={i} className="ev-quote">
                <span className="mono">p. {e.page}{e.column !== null ? ` col ${e.column + 1}` : ''}</span>
                {e.term && <> · {e.term}</>}: <q>{e.snippet}</q>
              </div>
            ))}
          </li>
        ))}
      </ul>
    </details>
  );
}

/** Screening tab: the proposed verdicts for the current criteria. The reviewer decides inclusion
 *  (Include / Exclude, with a required reason) for each publication, which may agree with or
 *  override the proposal, including an 'unsure' one; bulk decisions apply one reason to the
 *  selected rows. Every decision is one audited call; nothing here decides on its own. */
export function ScreeningPanel({ sessionId, screens, sources, criteria, disabled, run, onState }: {
  sessionId: string;
  screens: ScreenRow[];
  sources: EvidenceSource[];
  criteria: CriteriaConfig | null;
  disabled: boolean;
  run: Run;
  onState: OnState;
}) {
  const [reasons, setReasons] = useState<Record<string, string>>({});
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [bulkReason, setBulkReason] = useState('');
  const [showAll, setShowAll] = useState(false);
  const [status, setStatus] = useState('');
  const focusId = useRef<string | null>(null);              // where focus goes once controls re-enable
  const reasonRefs = useRef(new Map<string, HTMLInputElement>());
  const captionRef = useRef<HTMLTableCaptionElement>(null);

  const hash = criteria?.body_hash ?? null;
  const current = hash ? screens.filter(s => s.criteria_hash === hash) : screens;
  const older = screens.length - current.length;
  const rows = showAll ? screens : current;                 // ledger order: a decided row stays in place
  const texts = new Map((criteria?.body ?? []).map(c => [c.id, c.text]));
  const names = new Map(sources.map(s => [s.id, s]));
  const label = (s: ScreenRow) => names.get(s.source_id)?.filename ?? s.source_id;
  const proposed = rows.filter(s => s.status === 'proposed');
  const picked = proposed.filter(s => selected.has(s.id));

  // Keyboard users continue with the next undecided publication once the controls re-enable.
  useEffect(() => {
    const id = focusId.current;
    if (!id || disabled) return;
    if (id === 'caption') captionRef.current?.focus();
    else reasonRefs.current.get(id)?.focus();
    focusId.current = null;
  });

  const decide = async (decisions: ScreenDecision[]) => {
    const ok = await run(`Recording ${decisions.length} decision${decisions.length === 1 ? '' : 's'}…`, async () => {
      const res = await evidenceDecideScreen(sessionId, decisions);
      onState(res.state, res.summary);
    });
    if (!ok) return false;
    const done = new Set(decisions.map(d => d.screen_id));
    const left = proposed.filter(s => !done.has(s.id));
    const after = proposed.findIndex(s => done.has(s.id));
    const next = left.find((_, i) => i >= after) ?? left[0];
    const first = rows.find(s => s.id === decisions[0].screen_id);
    setStatus(`${decisions.length === 1 && first ? `${label(first)}: ${decisions[0].decision}d` : `${decisions.length} decisions recorded`}; `
      + `${left.length} publication${left.length === 1 ? '' : 's'} left to decide.`);
    focusId.current = next ? next.id : 'caption';
    setReasons(prev => Object.fromEntries(Object.entries(prev).filter(([k]) => !done.has(k))));
    setSelected(prev => new Set([...prev].filter(id => !done.has(id))));
    return true;
  };
  const one = (id: string, decision: Inclusion) =>
    void decide([{ screen_id: id, decision, reason: (reasons[id] ?? '').trim() }]);
  const bulk = (decision: Inclusion) =>
    void decide(picked.map(s => ({ screen_id: s.id, decision, reason: bulkReason.trim() })))
      .then(ok => { if (ok) setBulkReason(''); });
  const toggle = (id: string) => setSelected(prev => {
    const next = new Set(prev);
    if (next.has(id)) next.delete(id); else next.add(id);
    return next;
  });

  if (screens.length === 0) {
    return <p className="ev-empty">No screening verdicts yet. Save criteria and run screening from the Criteria tab.</p>;
  }
  const bulkWhy = picked.length === 0 ? 'Select undecided publications first' : !bulkReason.trim() ? 'A reason is required' : undefined;

  return (
    <div className="ev-panel">
      <p className="ev-intro">
        Each verdict is a proposal from <code>screen_publications</code>. You decide inclusion: Include or Exclude
        records your decision (and whether it agrees with the proposal) in the audit chain. Extraction reads only
        the publications you include.
      </p>
      <div className="ev-actions">
        <button type="button" className="btn btn-ghost" disabled={disabled || proposed.length === 0}
          onClick={() => setSelected(picked.length === proposed.length ? new Set() : new Set(proposed.map(s => s.id)))}>
          {picked.length === proposed.length && proposed.length > 0 ? 'Clear selection' : `Select all undecided (${proposed.length})`}
        </button>
        <input className="ev-input ev-bulk-reason" aria-label="Reason for the selected publications" value={bulkReason}
          maxLength={MAX_REASON} placeholder="Reason, applied to every selected publication" disabled={disabled}
          onChange={e => setBulkReason(e.target.value)} />
        <button type="button" className="btn btn-primary" disabled={disabled || !!bulkWhy} title={bulkWhy}
          onClick={() => bulk('include')}>
          Include selected ({picked.length})
        </button>
        <button type="button" className="btn btn-ghost" disabled={disabled || !!bulkWhy} title={bulkWhy}
          onClick={() => bulk('exclude')}>
          Exclude selected ({picked.length})
        </button>
        {older > 0 && (
          <button type="button" className="link-btn ev-push" onClick={() => setShowAll(v => !v)}>
            {showAll ? 'Show current criteria only' : `Show ${older} verdict${older === 1 ? '' : 's'} from earlier criteria`}
          </button>
        )}
      </div>
      <p className="sr-only" role="status">{status}</p>
      <div className="ev-table-wrap">
        <table className="ev-table ev-screen-table">
          <caption className="sr-only" ref={captionRef} tabIndex={-1}>Screening verdicts</caption>
          <thead>
            <tr>
              <th scope="col"><span className="sr-only">Select</span></th>
              <th scope="col">Publication</th><th scope="col" className="num">Score</th>
              <th scope="col">Proposed</th><th scope="col">Your decision</th>
            </tr>
          </thead>
          {rows.map(s => {
            const src = names.get(s.source_id);
            const reason = reasons[s.id] ?? '';
            const name = label(s);
            const inclusion = inclusionOf(s);
            return (
              <tbody key={s.id} className="ev-rowgroup">
                <tr>
                  <td>
                    {s.status === 'proposed' && (
                      <input type="checkbox" aria-label={`Select ${name}`} checked={selected.has(s.id)}
                        disabled={disabled} onChange={() => toggle(s.id)} />
                    )}
                  </td>
                  <th scope="row" className="ev-wide-cell ev-row-head">
                    <div className="ev-src-name">{name}</div>
                    {src?.title && <div className="ev-src-title">{src.title}</div>}
                    <Reasoning s={s} texts={s.criteria_hash === hash ? texts : new Map()} />
                    <div className="ev-prov">
                      <code>{s.id}</code> · {s.method === 'llm' ? 'model proposal' : 'keyword rules'}
                      {' · '}criteria <code>{shaPrefix(s.criteria_hash)}</code>{s.criteria_hash !== hash && ' (earlier)'}
                      {' · '}{hhmm(s.created_at)}
                    </div>
                  </th>
                  <td className="num mono">{s.score.toFixed(2)}</td>
                  <td><VerdictPill verdict={s.verdict} /></td>
                  <td>
                    {s.status === 'proposed' ? <span className="ev-dim">undecided</span> : (
                      <div className="ev-prov">
                        {inclusion ? <InclusionPill inclusion={inclusion} /> : <span className="ev-dim">{s.status}</span>}
                        <div>{s.status === 'approved' ? 'agrees with' : 'overrides'} the proposal · {s.decided_by} · {hhmm(s.decided_at)}</div>
                        <div className="ev-reason">“{s.decision_reason}”</div>
                      </div>
                    )}
                  </td>
                </tr>
                {s.status === 'proposed' && (
                  <tr className="ev-decide-row">
                    <td />
                    <td colSpan={4}>
                      <div className="ev-decide">
                        <input className="ev-input" ref={el => { if (el) reasonRefs.current.set(s.id, el); else reasonRefs.current.delete(s.id); }}
                          aria-label={`Reason for ${name}`} value={reason} maxLength={MAX_REASON}
                          placeholder="Reason (required)" disabled={disabled}
                          onChange={e => setReasons(prev => ({ ...prev, [s.id]: e.target.value }))} />
                        <button type="button" className="btn btn-primary" disabled={disabled || !reason.trim()}
                          aria-label={`Include ${name}`}
                          title={reason.trim() ? `Include this publication (proposed: ${s.verdict})` : 'A reason is required'}
                          onClick={() => one(s.id, 'include')}>Include</button>
                        <button type="button" className="btn btn-red" disabled={disabled || !reason.trim()}
                          aria-label={`Exclude ${name}`}
                          title={reason.trim() ? `Exclude this publication (proposed: ${s.verdict})` : 'A reason is required'}
                          onClick={() => one(s.id, 'exclude')}>Exclude</button>
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
