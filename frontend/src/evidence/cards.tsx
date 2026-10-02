/** Transcript result cards for the evidence tools. They render counts from the PharmState
 *  mirrors only (screening_results, extraction_results, evidence_summary,
 *  evidence_dataset_path); the rows themselves live in the Evidence workspace. */
import { hhmm } from '../shell/format';
import { shaPrefix } from './format';
import type { EvidenceSummary, ExtractionResults, ScreeningResults } from './types';

type Stat = { label: string; value: number | string };

function Strip({ stats }: { stats: Stat[] }) {
  return (
    <dl className="stat-strip">
      {stats.map(s => <div key={s.label} className="stat"><dt>{s.label}</dt><dd className="mono">{s.value}</dd></div>)}
    </dl>
  );
}

function Open({ onOpen, label }: { onOpen?: () => void; label: string }) {
  return onOpen ? <button type="button" className="link-btn" onClick={onOpen}>{label}</button> : null;
}

const methodWord = (m: string) => (m === 'llm' ? 'model' : m === 'keyword' ? 'keyword rules' : m === 'table' ? 'rules' : m);

export function EvidenceScreenCard({ r, onOpen }: { r: ScreeningResults; onOpen?: () => void }) {
  return (
    <div className="ev-card">
      <Strip stats={[
        { label: 'screened', value: r.n_screened },
        { label: 'include', value: r.verdicts.include },
        { label: 'exclude', value: r.verdicts.exclude },
        { label: 'unsure', value: r.verdicts.unsure },
        { label: 'awaiting you', value: r.proposed },
        { label: 'approved', value: r.approved },
        { label: 'rejected', value: r.rejected },
      ]} />
      <p className="ev-prov">
        Proposed verdicts · {methodWord(r.method)} · criteria <code>{shaPrefix(r.criteria_hash)}</code> · {hhmm(r.updated_at)}
        {r.proposed > 0 && ' · each verdict needs your decision before extraction reads the paper'}
      </p>
      <Open onOpen={onOpen} label="Review verdicts in the Evidence workspace" />
    </div>
  );
}

export function EvidenceRecordsCard({ r, onOpen }: { r: ExtractionResults; onOpen?: () => void }) {
  return (
    <div className="ev-card">
      <Strip stats={[
        { label: 'records', value: r.n_records },
        { label: 'awaiting you', value: r.proposed },
        { label: 'verify failed', value: r.verify_fail },
        { label: 'flagged', value: r.flagged },
        { label: 'approved', value: r.approved },
        { label: 'rejected', value: r.rejected },
      ]} />
      <p className="ev-prov">
        Proposed records · {methodWord(r.method)} · template <code>{shaPrefix(r.template_hash)}</code> · {hhmm(r.updated_at)}
        {r.verify_fail > 0 && ' · records that failed literal verification cannot be approved'}
      </p>
      <Open onOpen={onOpen} label="Review records in the Evidence workspace" />
    </div>
  );
}

export function EvidenceDatasetCard({ path, summary, onOpen }: {
  path: string; summary: EvidenceSummary | null; onOpen?: () => void;
}) {
  const name = path.split(/[\\/]/).pop() ?? path;
  return (
    <div className="ev-card">
      {summary && (
        <Strip stats={[
          { label: 'approved records', value: summary.records.approved },
          { label: 'publications', value: summary.sources },
          { label: 'exports', value: summary.datasets },
        ]} />
      )}
      <p className="ev-prov">
        Long CSV <code>{name}</code>{summary?.wide_path && <> · wide CSV <code>{summary.wide_path.split(/[\\/]/).pop()}</code></>}
        {summary?.template_hash && <> · template <code>{shaPrefix(summary.template_hash)}</code></>}
        {summary && <> · {hhmm(summary.updated_at)}</>}
      </p>
      <Open onOpen={onOpen} label="Download from the Evidence workspace" />
    </div>
  );
}
