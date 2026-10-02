import { useCallback, useEffect, useState } from 'react';
import { AlertTriangle } from 'lucide-react';
import { api } from '../api';
import { latestEntryFor } from '../shell/format';
import { ResultSection } from '../shell/ResultSection';
import type { AuditEntry, PharmState } from '../types';
import { evidenceSnapshot } from './api';
import { errorMessage, includedCount, latestConfigEntry, parsedCount } from './format';
import { CriteriaPanel } from './CriteriaPanel';
import { LibraryPanel } from './LibraryPanel';
import { TemplatePanel } from './TemplatePanel';
import type { EvidenceMarker, EvidenceSnapshot, OnTool, Run } from './types';
import './evidence.css';

const SECTION_ID = 'evidence-workspace';
type Tab = 'library' | 'criteria' | 'template';
const TABS: { key: Tab; label: string; title: string; tool: string }[] = [
  { key: 'library', label: 'Library', title: 'Publications', tool: 'ingest_evidence_source' },
  { key: 'criteria', label: 'Criteria', title: 'Screening criteria', tool: 'save_evidence_config' },
  { key: 'template', label: 'Template', title: 'Extraction template', tool: 'save_evidence_config' },
];

/** Evidence workspace: a tabbed document section over the session's evidence ledger.
 *  The snapshot (GET /evidence) is the source of truth and is re-read after every
 *  mutation; PharmState only carries summaries. Every tab shows the agent, tool and
 *  the audit seal of its latest call. Nothing here approves anything on its own. */
export function EvidenceWorkspace({ sessionId, audit, busy, onResult }: {
  sessionId: string;
  audit: AuditEntry[];
  busy: boolean;
  /** New PharmState (+ summary for the transcript, + result-card marker). */
  onResult: (state: PharmState, summary: string, marker?: EvidenceMarker) => void;
}) {
  const [tab, setTab] = useState<Tab>('library');
  const [snap, setSnap] = useState<EvidenceSnapshot | null>(null);
  const [version, setVersion] = useState(0);
  const [pending, setPending] = useState('');
  const [tick, setTick] = useState(0);
  const [error, setError] = useState('');

  // setState only after the await (react-hooks/set-state-in-effect); `alive`
  // drops a response that lands after a newer request or an unmount.
  useEffect(() => {
    let alive = true;
    const load = async () => {
      try {
        const s = await evidenceSnapshot(sessionId);
        if (alive) setSnap(s);
      } catch (e) {
        if (alive) setError(`Could not load the evidence ledger: ${errorMessage(e)}`);
      }
    };
    void load();
    return () => { alive = false; };
  }, [sessionId, version]);

  // Bring the section into view when the rail button opens it (no animation).
  useEffect(() => { document.getElementById(SECTION_ID)?.scrollIntoView({ block: 'start' }); }, []);

  const refresh = useCallback(() => setVersion(v => v + 1), []);

  /** Re-read PharmState after a call that returns none (upload, config save). */
  const syncState = useCallback(async () => {
    onResult(await api.getState(sessionId), '');
  }, [sessionId, onResult]);

  const run: Run = useCallback(async (label, task) => {
    setPending(label);
    setTick(0);
    setError('');
    try {
      await task(setTick);
      return true;
    } catch (e) {
      setError(errorMessage(e));
      return false;
    } finally {
      setPending('');
      refresh();
    }
  }, [refresh]);

  const onTool: OnTool = useCallback((env, marker) => onResult(env.state, env.summary, marker), [onResult]);

  const live = snap && snap.session_id === sessionId ? snap : null;
  const disabled = busy || !!pending || !live;
  const meta = TABS.find(t => t.key === tab) ?? TABS[0];
  const seal = tab === 'library' ? latestEntryFor(audit, meta.tool) : latestConfigEntry(audit, tab);

  const tabs = (
    <div className="seg" role="group" aria-label="Evidence step">
      {TABS.map(t => (
        <button key={t.key} type="button" aria-pressed={tab === t.key} onClick={() => setTab(t.key)}>
          {t.label}{t.key === 'library' && live ? ` (${live.sources.length})` : ''}
        </button>
      ))}
    </div>
  );

  return (
    <ResultSection id={SECTION_ID} title={`Evidence · ${meta.title}`} agent="evidence" tool={meta.tool}
      seal={seal?.entry_hash ?? null} controls={tabs}>
      <div className="ev-workspace">
        {live && (
          <p className="ev-counts">
            {live.counts.sources} publication{live.counts.sources === 1 ? '' : 's'} ({parsedCount(live)} parsed)
            {' · '}screening {live.counts.screens.proposed} proposed, {live.counts.screens.approved} approved
            {' · '}records {live.counts.records.proposed} proposed, {live.counts.records.approved} approved
            {' · '}{live.counts.datasets} export{live.counts.datasets === 1 ? '' : 's'}
          </p>
        )}
        {pending && (
          <div className="job-note" role="status">
            <div className="job-spinner" aria-hidden="true" /> {pending}{tick > 0 ? ` ${tick} s` : ''}
          </div>
        )}
        {error && (
          <div className="job-note ev-error" role="alert">
            <AlertTriangle size={13} aria-hidden="true" /> {error}
          </div>
        )}
        {!live && !error && <p className="ev-empty">Loading the evidence ledger…</p>}
        {/* All tabs stay mounted (hidden) so unsaved criteria/template edits survive a tab switch. */}
        {live && (
          <>
            <div hidden={tab !== 'library'}>
              <LibraryPanel sessionId={sessionId} sources={live.sources} disabled={disabled}
                onUploaded={() => { refresh(); syncState().catch(e => setError(errorMessage(e))); }} />
            </div>
            <div hidden={tab !== 'criteria'}>
              <CriteriaPanel key={live.criteria?.body_hash ?? 'none'} sessionId={sessionId} config={live.criteria}
                parsed={parsedCount(live)} screens={live.counts.screens} disabled={disabled}
                run={run} onSaved={syncState} onTool={onTool} />
            </div>
            <div hidden={tab !== 'template'}>
              <TemplatePanel key={live.template?.body_hash ?? 'none'} sessionId={sessionId} config={live.template}
                included={includedCount(live)} records={live.counts.records} disabled={disabled}
                run={run} onSaved={syncState} onTool={onTool} />
            </div>
          </>
        )}
      </div>
    </ResultSection>
  );
}
