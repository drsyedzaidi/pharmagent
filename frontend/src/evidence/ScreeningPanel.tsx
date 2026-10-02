import { useState } from 'react';
import { hhmm } from '../shell/format';
import { evidenceDecideScreen } from './api';
import { MetWord, StatusPill, VerdictPill } from './badges';
import { shaPrefix } from './format';
import type {
  CriteriaConfig, Decision, EvidenceSource, OnState, Run, ScreenDecision, ScreenRow,
} from './types';

const MAX_REASON = 500;
const STATUS_ORDER = { proposed: 0, approved: 1, rejected: 2 } as const;

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

/** Screening tab: the proposed verdicts for the current criteria, each with a required
 *  reason for Approve / Reject, plus bulk approval of selected proposals with one reason.
 *  Every decision is one audited call; nothing here decides on its own. */
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

  const hash = criteria?.body_hash ?? null;
  const current = hash ? screens.filter(s => s.criteria_hash === hash) : screens;
  const older = screens.length - current.length;
  const rows = [...(showAll ? screens : current)].sort((a, b) => STATUS_ORDER[a.status] - STATUS_ORDER[b.status]);
  const texts = new Map((criteria?.body ?? []).map(c => [c.id, c.text]));
  const names = new Map(sources.map(s => [s.id, s]));
  const proposed = rows.filter(s => s.status === 'proposed');
  const picked = proposed.filter(s => selected.has(s.id));

  const decide = async (decisions: ScreenDecision[]) => {
    const ok = await run(`Recording ${decisions.length} decision${decisions.length === 1 ? '' : 's'}…`, async () => {
      const res = await evidenceDecideScreen(sessionId, decisions);
      onState(res.state, res.summary);
    });
    if (!ok) return false;
    const done = new Set(decisions.map(d => d.screen_id));
    setReasons(prev => Object.fromEntries(Object.entries(prev).filter(([k]) => !done.has(k))));
    setSelected(prev => new Set([...prev].filter(id => !done.has(id))));
    return true;
  };
  const one = (id: string, decision: Decision) =>
    void decide([{ screen_id: id, decision, reason: (reasons[id] ?? '').trim() }]);
  const toggle = (id: string) => setSelected(prev => {
    const next = new Set(prev);
    if (next.has(id)) next.delete(id); else next.add(id);
    return next;
  });

  if (screens.length === 0) {
    return <p className="ev-empty">No screening verdicts yet. Save criteria and run screening from the Criteria tab.</p>;
  }

  return (
    <div className="ev-panel">
      <p className="ev-intro">
        Each verdict is a proposal from <code>screen_publications</code>. Approving it records your decision in the
        audit chain; extraction reads only publications whose include verdict you approved.
      </p>
      <div className="ev-actions">
        <button type="button" className="btn btn-ghost" disabled={disabled || proposed.length === 0}
          onClick={() => setSelected(picked.length === proposed.length ? new Set() : new Set(proposed.map(s => s.id)))}>
          {picked.length === proposed.length && proposed.length > 0 ? 'Clear selection' : `Select all proposed (${proposed.length})`}
        </button>
        <input className="ev-input ev-bulk-reason" aria-label="Reason for the selected verdicts" value={bulkReason}
          maxLength={MAX_REASON} placeholder="Reason, applied to every selected verdict" disabled={disabled}
          onChange={e => setBulkReason(e.target.value)} />
        <button type="button" className="btn btn-primary" disabled={disabled || picked.length === 0 || !bulkReason.trim()}
          title={picked.length === 0 ? 'Select proposed verdicts first' : !bulkReason.trim() ? 'A reason is required' : undefined}
          onClick={() => void decide(picked.map(s => ({ screen_id: s.id, decision: 'approve', reason: bulkReason.trim() })))
            .then(ok => { if (ok) setBulkReason(''); })}>
          Approve selected ({picked.length})
        </button>
        {older > 0 && (
          <button type="button" className="link-btn ev-push" onClick={() => setShowAll(v => !v)}>
            {showAll ? 'Show current criteria only' : `Show ${older} verdict${older === 1 ? '' : 's'} from earlier criteria`}
          </button>
        )}
      </div>
      <div className="ev-table-wrap">
        <table className="ev-table">
          <caption className="sr-only">Screening verdicts</caption>
          <thead>
            <tr>
              <th scope="col"><span className="sr-only">Select</span></th>
              <th scope="col">Publication</th><th scope="col" className="num">Score</th>
              <th scope="col">Verdict</th><th scope="col">Status</th><th scope="col">Decision</th>
            </tr>
          </thead>
          <tbody>
            {rows.map(s => {
              const src = names.get(s.source_id);
              const reason = reasons[s.id] ?? '';
              const label = src?.filename ?? s.source_id;
              return (
                <tr key={s.id}>
                  <td>
                    {s.status === 'proposed' && (
                      <input type="checkbox" aria-label={`Select verdict for ${label}`} checked={selected.has(s.id)}
                        disabled={disabled} onChange={() => toggle(s.id)} />
                    )}
                  </td>
                  <td className="ev-wide-cell">
                    <div className="ev-src-name">{label}</div>
                    {src?.title && <div className="ev-src-title">{src.title}</div>}
                    <Reasoning s={s} texts={s.criteria_hash === hash ? texts : new Map()} />
                    <div className="ev-prov">
                      <code>{s.id}</code> · {s.method === 'llm' ? 'model proposal' : 'keyword rules'}
                      {' · '}criteria <code>{shaPrefix(s.criteria_hash)}</code>{s.criteria_hash !== hash && ' (earlier)'}
                      {' · '}{hhmm(s.created_at)}
                    </div>
                  </td>
                  <td className="num mono">{s.score.toFixed(2)}</td>
                  <td><VerdictPill verdict={s.verdict} /></td>
                  <td><StatusPill status={s.status} /></td>
                  <td className="ev-decide-cell">
                    {s.status === 'proposed' ? (
                      <div className="ev-decide">
                        <input className="ev-input" aria-label={`Reason for ${label}`} value={reason} maxLength={MAX_REASON}
                          placeholder="Reason (required)" disabled={disabled}
                          onChange={e => setReasons(prev => ({ ...prev, [s.id]: e.target.value }))} />
                        <button type="button" className="btn btn-primary" disabled={disabled || !reason.trim()}
                          title={reason.trim() ? `Approve the ${s.verdict} verdict` : 'A reason is required'}
                          onClick={() => one(s.id, 'approve')}>Approve</button>
                        <button type="button" className="btn btn-red" disabled={disabled || !reason.trim()}
                          title={reason.trim() ? `Reject the ${s.verdict} verdict` : 'A reason is required'}
                          onClick={() => one(s.id, 'reject')}>Reject</button>
                      </div>
                    ) : (
                      <div className="ev-prov">
                        {s.status} by {s.decided_by} · {hhmm(s.decided_at)}
                        <div className="ev-reason">“{s.decision_reason}”</div>
                      </div>
                    )}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}
