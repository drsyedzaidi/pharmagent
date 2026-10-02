import { Download } from 'lucide-react';
import { hhmm } from '../shell/format';
import { evidenceDownload, evidenceExport } from './api';
import { shaPrefix } from './format';
import type { DatasetRow, EvidenceCounts, EvidenceSummary, OnTool, Run } from './types';

const basename = (p: string) => p.split(/[\\/]/).pop() ?? p;

/** Dataset tab: ledger counts, Export (approved records only) and the session's exports
 *  with both CSVs downloadable and their sha256 shown in mono. */
export function DatasetPanel({ sessionId, summary, counts, datasets, disabled, run, onTool }: {
  sessionId: string;
  summary: EvidenceSummary | null;
  counts: EvidenceCounts;
  datasets: DatasetRow[];
  disabled: boolean;
  run: Run;
  onTool: OnTool;
}) {
  const approved = counts.records.approved;
  const stats: { label: string; value: number }[] = [
    { label: 'publications', value: summary?.sources ?? counts.sources },
    { label: 'parsed', value: summary?.parsed ?? 0 },
    { label: 'verdicts approved', value: counts.screens.approved },
    { label: 'records approved', value: approved },
    { label: 'records awaiting you', value: counts.records.proposed },
    { label: 'exports', value: counts.datasets },
  ];
  const latest = [...datasets].sort((a, b) => b.exported_at.localeCompare(a.exported_at));
  const why = approved === 0 ? 'Approve at least one record first.' : '';

  const doExport = () => run('Exporting the evidence dataset…', async () => {
    onTool(await evidenceExport(sessionId), '__EVIDENCE_DATASET__');
  });
  const get = (path: string) => run(`Downloading ${basename(path)}…`, async () => {
    await evidenceDownload(sessionId, path);
  });

  return (
    <div className="ev-panel">
      <dl className="stat-strip">
        {stats.map(s => (
          <div key={s.label} className="stat"><dt>{s.label}</dt><dd className="mono">{s.value}</dd></div>
        ))}
      </dl>
      <div className="ev-run">
        <button type="button" className={`btn ${why ? 'btn-ghost' : 'btn-primary'}`} disabled={disabled || !!why}
          title={why || 'Write the long and wide CSVs of every approved record'} onClick={() => void doExport()}>
          Export dataset
        </button>
        <span className="ev-hint">
          {why || `${approved} approved record${approved === 1 ? '' : 's'} will be written; proposed and rejected records are left out.`}
        </span>
      </div>
      {latest.length === 0 ? (
        <p className="ev-empty">No exports yet.</p>
      ) : (
        <ol className="ev-datasets" aria-label="Exports">
          {latest.map(d => (
            <li key={d.id} className="ev-dataset">
              <div className="ev-dataset-head">
                <span className="ev-src-name">{d.n_records} record{d.n_records === 1 ? '' : 's'} from {d.n_sources} publication{d.n_sources === 1 ? '' : 's'}</span>
                <span className="ev-prov">
                  <code>{d.id}</code> · by {d.exported_by} · {hhmm(d.exported_at)} · template <code>{shaPrefix(d.template_hash)}</code>
                </span>
              </div>
              {([['Long', d.long_path, d.long_sha256], ['Wide', d.wide_path, d.wide_sha256]] as const).map(([label, path, sha]) => (
                <div key={label} className="ev-file">
                  <button type="button" className="btn btn-ghost" disabled={disabled} onClick={() => void get(path)}>
                    <Download size={14} aria-hidden="true" /> {label} CSV
                  </button>
                  <span className="mono ev-filename">{basename(path)}</span>
                  <span className="ev-prov ev-sha">sha256 <code>{sha}</code></span>
                </div>
              ))}
            </li>
          ))}
        </ol>
      )}
    </div>
  );
}
