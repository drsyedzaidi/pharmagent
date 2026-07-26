import { useEffect, useRef, useState } from 'react';
import { api } from '../api';
import type { VariableMeta, FlexplotSpec, FlexplotData } from '../types';
import { VariablePicker } from './VariablePicker';
import { GraphicOptions } from './GraphicOptions';
import { FlexplotChart, type ChartOpts } from './FlexplotChart';
import './flexplot.css';

const DEFAULT_OPTS: ChartOpts = { logY: false, alpha: 0.7, showCI: true };

/** Pick sensible starting variables from detected NONMEM roles, else fall back. */
function seedSpec(vars: VariableMeta[], roles: Record<string, string>): FlexplotSpec | null {
  if (vars.length === 0) return null;
  const byRole = (r: string) => Object.keys(roles).find(k => roles[k] === r);
  const continuous = vars.filter(v => v.type === 'continuous');
  const y = byRole('DV') ?? continuous[0]?.name ?? vars[0].name;
  const x = byRole('TIME') ?? continuous.find(v => v.name !== y)?.name ?? null;
  return {
    y, x, color_by: null, panel_by: null,
    fit: 'loess', geom: 'points', center: 'median_iqr',
    ghost: false, log_y: false, jitter: 0.2, n_bins: 10, ci: 0.95,
  };
}

export function FlexplotPanel({ sessionId }: { sessionId: string }) {
  const [variables, setVariables] = useState<VariableMeta[]>([]);
  const [spec, setSpec] = useState<FlexplotSpec | null>(null);
  const [opts, setOpts] = useState<ChartOpts>(DEFAULT_OPTS);
  const [data, setData] = useState<FlexplotData | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const reqId = useRef(0);

  // Load the variable list and seed the initial selection. Every setState below
  // sits inside an async callback: setting state synchronously in an effect body
  // triggers a cascading render (react-hooks/set-state-in-effect), so the
  // loading/error resets live in the promise handlers and in patchSpec instead.
  useEffect(() => {
    let cancelled = false;
    api.variables(sessionId)
      .then(res => {
        if (cancelled) return;
        setError('');
        setVariables(res.variables);
        const seeded = seedSpec(res.variables, res.detected_roles);
        // Seeding a usable spec starts the plot fetch below, so enter the loading
        // state with it — but only when that fetch will actually run, or the
        // spinner would never clear.
        if (seeded?.y) setLoading(true);
        setSpec(seeded);
      })
      .catch(e => { if (!cancelled) setError((e as Error).message); });
    return () => { cancelled = true; };
  }, [sessionId]);

  // Rebuild the plot whenever a fetch-affecting option changes; a request-id
  // guard drops stale responses so fast edits never render out of order.
  useEffect(() => {
    if (!spec?.y) return;
    const id = ++reqId.current;
    api.flexplot(sessionId, spec)
      .then(res => {
        if (id !== reqId.current) return;
        setError('');
        setData(res.state.flexplot_data);
      })
      .catch(e => { if (id === reqId.current) setError((e as Error).message); })
      .finally(() => { if (id === reqId.current) setLoading(false); });
  }, [sessionId, spec]);

  const patchSpec = (patch: Partial<FlexplotSpec>) => {
    // Changing the spec is what triggers the refetch, so the loading state is
    // owned by this event handler rather than by the effect that reacts to it.
    if (spec) {
      setLoading(true);
      setError('');
    }
    setSpec(s => {
      if (!s) return s;
      const next = { ...s, ...patch };
      // A variable may occupy only one slot: clear any downstream slot that now
      // collides with an upstream selection, so a contradictory spec (e.g.
      // x === y, or color_by === x) is never sent to the server.
      if (next.x === next.y) next.x = null;
      if (next.color_by && (next.color_by === next.y || next.color_by === next.x)) next.color_by = null;
      if (next.panel_by && (next.panel_by === next.y || next.panel_by === next.x)) next.panel_by = null;
      return next;
    });
  };
  const patchOpts = (patch: Partial<ChartOpts>) =>
    setOpts(o => ({ ...o, ...patch }));

  if (variables.length === 0 && !error) {
    return <div className="flexplot-panel flexplot-status">Loading variables…</div>;
  }

  return (
    <div className="flexplot-panel">
      <div className="flexplot-controls">
        {spec && (
          <VariablePicker variables={variables} spec={spec} onChange={patchSpec} disabled={loading} />
        )}
        {spec && (
          <GraphicOptions
            spec={spec} onSpecChange={patchSpec}
            opts={opts} onOptsChange={patchOpts}
            kind={data?.kind ?? null} disabled={loading}
          />
        )}
      </div>

      {error && <div className="flexplot-error">{error}</div>}

      <div className="flexplot-canvas" aria-busy={loading}>
        {data ? <FlexplotChart data={data} opts={opts} />
              : !error && <div className="flexplot-status">Building plot…</div>}
      </div>
    </div>
  );
}
