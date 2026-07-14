import type { VariableMeta, FlexplotSpec } from '../types';

type Props = {
  variables: VariableMeta[];
  spec: FlexplotSpec;
  onChange: (patch: Partial<FlexplotSpec>) => void;
  disabled: boolean;
};

const NONE = '';

/** The four variable slots mirroring the jamovi flexplot panel. */
export function VariablePicker({ variables, spec, onChange, disabled }: Props) {
  const opt = (v: VariableMeta) => (
    <option key={v.name} value={v.name}>
      {v.name}{v.type === 'categorical' ? ' (cat)' : ''}
    </option>
  );

  return (
    <div className="flexplot-picker">
      <Field label="Outcome variable">
        <select className="model-select" value={spec.y} disabled={disabled}
                onChange={e => onChange({ y: e.target.value })}>
          {variables.map(opt)}
        </select>
      </Field>

      <Field label="Predictor variable">
        <select className="model-select" value={spec.x ?? NONE} disabled={disabled}
                onChange={e => onChange({ x: e.target.value || null })}>
          <option value={NONE}>— none (distribution) —</option>
          {variables.filter(v => v.name !== spec.y).map(opt)}
        </select>
      </Field>

      <Field label="Color / group by">
        {/* Offer every column except the axes: numeric-coded groups (e.g. DOSE)
            look continuous in metadata but the server classifies them as
            categorical, and >6 levels drop the aesthetic gracefully. */}
        <select className="model-select" value={spec.color_by ?? NONE} disabled={disabled}
                onChange={e => onChange({ color_by: e.target.value || null })}>
          <option value={NONE}>— none —</option>
          {variables.filter(v => v.name !== spec.y && v.name !== spec.x).map(opt)}
        </select>
      </Field>

      <Field label="Paneled variable">
        <select className="model-select" value={spec.panel_by ?? NONE} disabled={disabled}
                onChange={e => onChange({ panel_by: e.target.value || null })}>
          <option value={NONE}>— none —</option>
          {variables.filter(v => v.name !== spec.y && v.name !== spec.x).map(opt)}
        </select>
      </Field>
    </div>
  );
}

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <label className="flexplot-field">
      <span className="flexplot-field-label">{label}</span>
      {children}
    </label>
  );
}
