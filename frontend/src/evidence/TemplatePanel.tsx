import { useState } from 'react';
import { AlertTriangle, Plus, X } from 'lucide-react';
import { hhmm } from '../shell/format';
import { evidenceExtract, evidenceTemplate } from './api';
import { blankField, fromFieldDraft, sameBody, shaPrefix, templateProblem, toFieldDraft } from './format';
import type { FieldDraft } from './format';
import type { DecimalSetting, ExtractResult, ExtractSourceRow, FieldType, OnTool, Run, StatusCounts, TemplateConfig } from './types';

const TYPES: FieldType[] = ['number', 'integer', 'text'];
const DECIMALS: { v: DecimalSetting; label: string }[] = [
  { v: 'auto', label: 'Decimal: auto' }, { v: '.', label: 'Decimal: point' }, { v: ',', label: 'Decimal: comma' },
];

/** Template tab: the fields to extract (name, type, unit, range, aliases), Save
 *  (hash in mono) and Run extraction. Keyed by the saved body_hash like CriteriaPanel. */
export function TemplatePanel({ sessionId, config, included, records, disabled, run, onSaved, onTool }: {
  sessionId: string;
  config: TemplateConfig | null;
  /** Publications with an approved 'include' verdict. */
  included: number;
  records: StatusCounts;
  disabled: boolean;
  run: Run;
  onSaved: () => Promise<void>;
  onTool: OnTool;
}) {
  const saved = config ? config.body.fields.map(toFieldDraft) : [];
  const [rows, setRows] = useState<FieldDraft[]>(() => (saved.length ? saved : [blankField()]));
  /** Per-source outcome of the last extraction run from this tab (unmatched labels, model fallback). */
  const [lastRun, setLastRun] = useState<ExtractSourceRow[] | null>(null);
  const problem = templateProblem(rows);
  const dirty = !config || !sameBody(rows.map(fromFieldDraft), saved.map(fromFieldDraft));

  const edit = (i: number, p: Partial<FieldDraft>) =>
    setRows(prev => prev.map((r, j) => (j === i ? { ...r, ...p } : r)));

  const save = () => run('Saving template…', async () => {
    await evidenceTemplate(sessionId, { fields: rows.map(fromFieldDraft) });
    await onSaved();
  });
  const extract = () => run('Extracting records…', async tick => {
    const env = await evidenceExtract(sessionId, undefined, tick);
    setLastRun((env.result as ExtractResult).sources ?? []);
    onTool(env, '__EVIDENCE_RECORDS__');
  });
  const unmatched = [...new Set((lastRun ?? []).flatMap(s => s.unmatched_headers))];
  const fellBack = (lastRun ?? []).filter(s => s.llm_error);

  const why = !config ? 'Save the template first.'
    : dirty ? 'Save or discard your edits first.'
    : included === 0 ? 'Include at least one publication on the Screening tab first.' : '';

  return (
    <div className="ev-panel">
      <p className="ev-intro">
        Fields to extract from the included publications. Aliases are the header or row labels a table may use,
        exactly as printed and separated by semicolons (a label may contain a comma, e.g. Cmax,ss). Every proposed
        value must be found literally on its cited page.
      </p>
      <ol className="ev-rows" aria-label="Extraction template fields">
        {rows.map((r, i) => (
          <li key={i} className="ev-row ev-field-row">
            <input className="ev-input mono" aria-label={`Field ${i + 1} name`} value={r.name} maxLength={64}
              placeholder="cmax" disabled={disabled} onChange={e => edit(i, { name: e.target.value })} />
            <select className="ev-input" aria-label={`Field ${i + 1} type`} value={r.type} disabled={disabled}
              onChange={e => edit(i, { type: TYPES.find(t => t === e.target.value) ?? 'number' })}>
              {TYPES.map(t => <option key={t} value={t}>{t}</option>)}
            </select>
            <input className="ev-input" aria-label={`Field ${i + 1} unit`} value={r.unit} placeholder="Unit"
              disabled={disabled} onChange={e => edit(i, { unit: e.target.value })} />
            <input className="ev-input num" aria-label={`Field ${i + 1} minimum`} value={r.min} placeholder="Min"
              inputMode="decimal" disabled={disabled || r.type === 'text'} onChange={e => edit(i, { min: e.target.value })} />
            <input className="ev-input num" aria-label={`Field ${i + 1} maximum`} value={r.max} placeholder="Max"
              inputMode="decimal" disabled={disabled || r.type === 'text'} onChange={e => edit(i, { max: e.target.value })} />
            <button type="button" className="btn btn-ghost ev-icon-btn" aria-label={`Remove field ${i + 1}`}
              disabled={disabled || rows.length === 1} onClick={() => setRows(prev => prev.filter((_, j) => j !== i))}>
              <X size={14} aria-hidden="true" />
            </button>
            <label className="ev-check">
              <input type="checkbox" checked={r.required} disabled={disabled}
                onChange={e => edit(i, { required: e.target.checked })} /> Required
            </label>
            <select className="ev-input" aria-label={`Field ${i + 1} decimal separator`} value={r.decimal} disabled={disabled}
              onChange={e => edit(i, { decimal: DECIMALS.find(d => d.v === e.target.value)?.v ?? 'auto' })}>
              {DECIMALS.map(d => <option key={d.v} value={d.v}>{d.label}</option>)}
            </select>
            <input className="ev-input ev-span-wide" aria-label={`Field ${i + 1} aliases`} value={r.aliases}
              placeholder="Aliases, separated by ; (e.g. Cmax; Cmax,ss; Peak concentration)" disabled={disabled}
              onChange={e => edit(i, { aliases: e.target.value })} />
            <input className="ev-input ev-span-all" aria-label={`Field ${i + 1} description`} value={r.description}
              placeholder="Description (optional)" maxLength={500} disabled={disabled}
              onChange={e => edit(i, { description: e.target.value })} />
          </li>
        ))}
      </ol>
      <div className="ev-actions">
        <button type="button" className="btn btn-ghost" disabled={disabled || rows.length >= 60}
          onClick={() => setRows(prev => [...prev, blankField()])}>
          <Plus size={14} aria-hidden="true" /> Add field
        </button>
        {dirty && config && (
          <button type="button" className="link-btn" disabled={disabled} onClick={() => setRows(saved)}>Discard edits</button>
        )}
        <button type="button" className={`btn ${dirty ? 'btn-primary' : 'btn-ghost'} ev-push`}
          disabled={disabled || !dirty || !!problem} onClick={() => void save()}>
          Save template
        </button>
      </div>
      {problem && <p className="ev-problem" role="status"><AlertTriangle size={13} aria-hidden="true" /> {problem}</p>}
      <p className="ev-prov">
        {config
          ? <>Saved template · hash <code>{shaPrefix(config.body_hash)}</code> · by {config.saved_by} · {hhmm(config.saved_at)}{dirty && ' · unsaved changes'}</>
          : 'Not saved yet.'}
      </p>

      <div className="ev-run">
        <button type="button" className={`btn ${why ? 'btn-ghost' : 'btn-primary'}`} disabled={disabled || !!why}
          title={why || 'Propose records from the included publications; each is verified against its page'}
          onClick={() => void extract()}>
          Run extraction
        </button>
        <span className="ev-hint">
          {why || `${included} included publication${included === 1 ? '' : 's'} will be read.`}
          {' '}Records so far: {records.proposed} proposed · {records.approved} approved · {records.rejected} rejected.
        </span>
      </div>
      {fellBack.length > 0 && (
        <p className="ev-problem" role="status">
          <AlertTriangle size={13} aria-hidden="true" />
          The model failed on {fellBack.length} publication{fellBack.length === 1 ? '' : 's'}; the keyless table rules
          were used instead ({fellBack[0].llm_error}).
        </p>
      )}
      {unmatched.length > 0 && (
        <div className="ev-unmatched">
          <p className="ev-hint">
            Table labels the last run did not match to any field. Add the ones you want as aliases, exactly as printed:
          </p>
          <ul className="ev-chip-list" aria-label="Unmatched table labels">
            {unmatched.map(h => <li key={h}><code>{h}</code></li>)}
          </ul>
        </div>
      )}
    </div>
  );
}
