import { useState } from 'react';
import { AlertTriangle, Plus, X } from 'lucide-react';
import { hhmm } from '../shell/format';
import { evidenceCriteria, evidenceScreen } from './api';
import {
  blankCriterion, criteriaProblem, fromCriterionDraft, sameBody, shaPrefix, toCriterionDraft,
} from './format';
import type { CriterionDraft } from './format';
import type { CriteriaConfig, OnTool, Run, StatusCounts } from './types';

/** Criteria tab: editable inclusion/exclusion rows, Save (hash shown in mono) and
 *  Run screening. Mounted with key = saved body_hash, so a save or reload reseeds
 *  the local rows without an effect. */
export function CriteriaPanel({ sessionId, config, parsed, screens, disabled, run, onSaved, onTool }: {
  sessionId: string;
  config: CriteriaConfig | null;
  parsed: number;
  screens: StatusCounts;
  disabled: boolean;
  run: Run;
  onSaved: () => Promise<void>;
  onTool: OnTool;
}) {
  const saved = config ? config.body.map(toCriterionDraft) : [];
  const [rows, setRows] = useState<CriterionDraft[]>(() => (saved.length ? saved : [blankCriterion([])]));
  const problem = criteriaProblem(rows);
  const dirty = !config || !sameBody(rows.map(fromCriterionDraft), saved.map(fromCriterionDraft));

  const edit = (i: number, p: Partial<CriterionDraft>) =>
    setRows(prev => prev.map((r, j) => (j === i ? { ...r, ...p } : r)));
  const remove = (i: number) => setRows(prev => prev.filter((_, j) => j !== i));

  const save = () => run('Saving criteria…', async () => {
    await evidenceCriteria(sessionId, rows.map(fromCriterionDraft));
    await onSaved();
  });
  const screen = () => run('Screening publications…', async tick => {
    const env = await evidenceScreen(sessionId, undefined, tick);
    onTool(env, '__EVIDENCE_SCREEN__');
  });

  const why = parsed === 0 ? 'Upload at least one parsed PDF first.'
    : !config ? 'Save the criteria first.'
    : dirty ? 'Save or discard your edits first.' : '';

  return (
    <div className="ev-panel">
      <p className="ev-intro">
        Each criterion is checked against the first pages of every parsed publication. Without a model the check
        is keyword matching on the listed keywords (or the criterion text); verdicts are proposals you decide.
      </p>
      <ol className="ev-rows" aria-label="Screening criteria">
        {rows.map((r, i) => (
          <li key={i} className="ev-row ev-crit-row">
            <input className="ev-input mono" aria-label={`Criterion ${i + 1} id`} value={r.id} maxLength={64}
              disabled={disabled} onChange={e => edit(i, { id: e.target.value })} />
            <select className="ev-input" aria-label={`Criterion ${i + 1} kind`} value={r.kind} disabled={disabled}
              onChange={e => edit(i, { kind: e.target.value === 'exclusion' ? 'exclusion' : 'inclusion' })}>
              <option value="inclusion">Inclusion</option>
              <option value="exclusion">Exclusion</option>
            </select>
            <input className="ev-input" aria-label={`Criterion ${i + 1} text`} value={r.text} maxLength={500}
              placeholder="e.g. Reports human plasma pharmacokinetics" disabled={disabled}
              onChange={e => edit(i, { text: e.target.value })} />
            <button type="button" className="btn btn-ghost ev-icon-btn" aria-label={`Remove criterion ${i + 1}`}
              disabled={disabled || rows.length === 1} onClick={() => remove(i)}>
              <X size={14} aria-hidden="true" />
            </button>
            <label className="ev-check">
              <input type="checkbox" checked={r.required} disabled={disabled}
                onChange={e => edit(i, { required: e.target.checked })} /> Required
            </label>
            <input className="ev-input ev-span" aria-label={`Criterion ${i + 1} keywords`} value={r.keywords}
              placeholder="Keywords, comma separated (optional)" disabled={disabled}
              onChange={e => edit(i, { keywords: e.target.value })} />
          </li>
        ))}
      </ol>
      <div className="ev-actions">
        <button type="button" className="btn btn-ghost" disabled={disabled || rows.length >= 100}
          onClick={() => setRows(prev => [...prev, blankCriterion(prev)])}>
          <Plus size={14} aria-hidden="true" /> Add criterion
        </button>
        {dirty && config && (
          <button type="button" className="link-btn" disabled={disabled} onClick={() => setRows(saved)}>Discard edits</button>
        )}
        <button type="button" className={`btn ${dirty ? 'btn-primary' : 'btn-ghost'} ev-push`}
          disabled={disabled || !dirty || !!problem} onClick={() => void save()}>
          Save criteria
        </button>
      </div>
      {problem && <p className="ev-problem" role="status"><AlertTriangle size={13} aria-hidden="true" /> {problem}</p>}
      <p className="ev-prov">
        {config
          ? <>Saved criteria · hash <code>{shaPrefix(config.body_hash)}</code> · by {config.saved_by} · {hhmm(config.saved_at)}{dirty && ' · unsaved changes'}</>
          : 'Not saved yet.'}
      </p>

      <div className="ev-run">
        <button type="button" className={`btn ${why ? 'btn-ghost' : 'btn-primary'}`} disabled={disabled || !!why}
          title={why || 'Propose include / exclude / unsure verdicts for every parsed publication'} onClick={() => void screen()}>
          Run screening
        </button>
        <span className="ev-hint">
          {why || `${parsed} parsed publication${parsed === 1 ? '' : 's'} will be screened.`}
          {' '}Verdicts so far: {screens.proposed} proposed · {screens.approved} approved · {screens.rejected} rejected.
        </span>
      </div>
    </div>
  );
}
