import { useCallback, useEffect, useRef, useState } from 'react';
import { AlertTriangle } from 'lucide-react';
import { api } from '../api';
import { latestEntryFor } from '../shell/format';
import { ResultSection } from '../shell/ResultSection';
import type { AuditEntry, PharmState } from '../types';
import { evidenceSnapshot } from './api';
import { errorMessage, includedCount, latestConfigEntry, parsedCount } from './format';
import { CriteriaPanel } from './CriteriaPanel';
import { DatasetPanel } from './DatasetPanel';
import { LibraryPanel } from './LibraryPanel';
import { RecordsPanel } from './RecordsPanel';
import { ScreeningPanel } from './ScreeningPanel';
import { TemplatePanel } from './TemplatePanel';
import { useRelayout } from './useRelayout';
import type { EvidenceFocus, EvidenceMarker, EvidenceSnapshot, EvidenceTab, OnTool, Run } from './types';
import './evidence.css';

const SECTION_ID = 'evidence-workspace';
type Tab = EvidenceTab;
const TABS: { key: Tab; label: string; title: string; tool: string }[] = [
  { key: 'library', label: 'Library', title: 'Publications', tool: 'ingest_evidence_source' },
  { key: 'criteria', label: 'Criteria', title: 'Screening criteria', tool: 'save_evidence_config' },
  { key: 'screening', label: 'Screening', title: 'Screening verdicts', tool: 'screen_publications' },
  { key: 'template', label: 'Template', title: 'Extraction template', tool: 'save_evidence_config' },
  { key: 'records', label: 'Records', title: 'Extracted records', tool: 'extract_evidence' },
  { key: 'dataset', label: 'Dataset', title: 'Evidence dataset', tool: 'export_evidence_dataset' },
];
/** Pending-work count shown in a tab label, so decisions waiting elsewhere stay visible. */
const badge = (snap: EvidenceSnapshot, t: Tab): string => {
  const n = t === 'library' ? snap.sources.length
    : t === 'screening' ? snap.counts.screens.proposed
    : t === 'records' ? snap.counts.records.proposed
    : t === 'dataset' ? snap.counts.datasets : 0;
  return t === 'library' || n > 0 ? ` (${n})` : '';
};

/** Evidence workspace: a tabbed document section over the session's evidence ledger.
 *  The snapshot (GET /evidence) is the source of truth and is re-read after every
 *  mutation; PharmState only carries summaries. Every tab shows the agent, tool and
 *  the audit seal of its latest call. Nothing here approves anything on its own. */
export function EvidenceWorkspace({ sessionId, audit, busy, focus = null, stamp = null, onResult }: {
  sessionId: string;
  /** PharmState evidence_summary.updated_at: a change made elsewhere (a chat turn) re-reads the ledger. */
  stamp?: string | null;
  audit: AuditEntry[];
  busy: boolean;
  /** Tab to show when opened from a chat card; a new object re-focuses an open workspace. */
  focus?: EvidenceFocus | null;
  /** New PharmState (+ summary for the transcript, + result-card marker). */
  onResult: (state: PharmState, summary: string, marker?: EvidenceMarker) => void;
}) {
  const [tab, setTab] = useState<Tab>(focus?.tab ?? 'library');
  const [seenFocus, setSeenFocus] = useState(focus);
  if (focus !== seenFocus) {           // derive-from-props: a card asked for a tab
    setSeenFocus(focus);
    if (focus) setTab(focus.tab);
  }
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
  }, [sessionId, version, stamp]);

  // Mounting and a new focus request are both user actions (the rail button, a chat card's link):
  // bring the section into view (no animation) and move keyboard focus to the current step.
  const tabsRef = useRef<HTMLDivElement>(null);
  useEffect(() => {
    document.getElementById(SECTION_ID)?.scrollIntoView({ block: 'start' });
    tabsRef.current?.querySelector<HTMLButtonElement>('button[aria-pressed="true"]')?.focus({ preventScroll: true });
  }, [focus]);

  const refresh = useCallback(() => setVersion(v => v + 1), []);

  /** Re-read PharmState after a call that returns none (upload, config save). */
  const syncState = useCallback(async () => {
    onResult(await api.getState(sessionId), '');
  }, [sessionId, onResult]);

  /** A layout rebuild settled: re-read the ledger (layout status) and PharmState (its audit entry). */
  const onLayoutChanged = useCallback(() => {
    refresh();
    syncState().catch(e => setError(errorMessage(e)));
  }, [refresh, syncState]);

  // One set of rebuild controls for the Library list and every open record's notice: one in-flight guard.
  const relayout = useRelayout(sessionId, onLayoutChanged);

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

  const onTool: OnTool = useCallback((env, marker) => {
    onResult(env.state, env.summary, marker);
    if (marker === '__EVIDENCE_SCREEN__') setTab('screening');
    else if (marker === '__EVIDENCE_RECORDS__') setTab('records');
  }, [onResult]);

  const live = snap && snap.session_id === sessionId ? snap : null;
  const disabled = busy || !!pending || !live;
  const meta = TABS.find(t => t.key === tab) ?? TABS[0];
  const seal = tab === 'criteria' || tab === 'template' ? latestConfigEntry(audit, tab) : latestEntryFor(audit, meta.tool);

  const tabs = (
    <div className="seg" role="group" aria-label="Evidence step" ref={tabsRef}>
      {TABS.map(t => (
        <button key={t.key} type="button" aria-pressed={tab === t.key} onClick={() => setTab(t.key)}>
          {t.label}{live ? badge(live, t.key) : ''}
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
            {' · '}screening {live.counts.screens.proposed} undecided, {includedCount(live)} included
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
                relayout={relayout}
                onUploaded={() => { refresh(); syncState().catch(e => setError(errorMessage(e))); }} />
            </div>
            <div hidden={tab !== 'criteria'}>
              <CriteriaPanel key={live.criteria?.body_hash ?? 'none'} sessionId={sessionId} config={live.criteria}
                parsed={parsedCount(live)} screens={live.counts.screens} disabled={disabled}
                run={run} onSaved={syncState} onTool={onTool} />
            </div>
            <div hidden={tab !== 'screening'}>
              <ScreeningPanel sessionId={sessionId} screens={live.screens} sources={live.sources} criteria={live.criteria}
                disabled={disabled} run={run} onState={onResult} />
            </div>
            <div hidden={tab !== 'template'}>
              <TemplatePanel key={live.template?.body_hash ?? 'none'} sessionId={sessionId} config={live.template}
                included={includedCount(live)} records={live.counts.records} disabled={disabled}
                run={run} onSaved={syncState} onTool={onTool} />
            </div>
            <div hidden={tab !== 'records'}>
              <RecordsPanel sessionId={sessionId} records={live.records} sources={live.sources} template={live.template}
                disabled={disabled} run={run} onState={onResult} onTool={onTool} relayout={relayout} />
            </div>
            <div hidden={tab !== 'dataset'}>
              <DatasetPanel sessionId={sessionId} summary={live.summary} counts={live.counts} datasets={live.datasets}
                disabled={disabled} run={run} onTool={onTool} />
            </div>
          </>
        )}
      </div>
    </ResultSection>
  );
}
