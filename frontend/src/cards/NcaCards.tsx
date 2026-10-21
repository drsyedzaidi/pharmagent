import { useEffect, useMemo, useState } from 'react';
import { AlertTriangle, CheckCircle, Maximize2 } from 'lucide-react';
import { api } from '../api';
import { QcEvidence } from '../shell/QcEvidence';
import type {
  LlmConfig,
  LlmProvider,
  LzSubject,
  NcaPlotData,
  PharmState,
  ReviewFinding,
  ReviewResults,
  Severity,
  SkillDef,
  SpaghettiData,
} from '../types';
import {
  ExtrapBadge,
  XRangeSlider,
  ZoomModal,
  Zoomable,
} from './shared';
import {
  AXIS_TITLE,
  EXTRAP_LIMIT,
  NCA_ROWS_COLLAPSED,
  NCA_ROWS_THRESHOLD,
  SPAG_PALETTE,
  TICK_TEXT,
  bySubjectId,
  errorText,
  flaggedSentence,
  fmt,
  labelUnit,
  ncaUnits,
  niceTicks,
  unitSuffix,
} from './helpers';

function QcCard({ state, onReviewLz, onReviewInPanel }: { state: PharmState; onReviewLz?: () => void; onReviewInPanel?: () => void }) {
  const v = state.qc_verdict ?? '';
  const cls = v === 'PASS' ? 'pass' : v.includes('CONDITIONAL') ? 'conditional' : 'fail';
  // One verdict line; the checklist itself lives in the review panel.
  return (
    <div className={`qc-card ${cls}`}>
      <QcEvidence verdict={v} checklist={state.qc_checklist} issues={state.qc_issues} subjects={state.nca_parameters}
        onReviewLz={onReviewLz} onReviewInPanel={onReviewInPanel} compact />
    </div>
  );
}

const SEVERITY_COLOR: Record<Severity, string> = {
  CRITICAL: '#B23A2E', HIGH: '#9A5B12', MEDIUM: '#1F66A6', LOW: '#536A83',
};

function ReviewCard({ r }: { r: ReviewResults }) {
  const c = r.counts;
  return (
    <div className="review-card">
      <div className="review-goal"
        style={{ color: r.goal_met ? '#1D7A5A' : '#9A5B12', fontWeight: 600 }}>
        {r.goal_met
          ? <><CheckCircle size={14} style={{ display: 'inline', marginRight: 6 }} />Goal met</>
          : <><AlertTriangle size={14} style={{ display: 'inline', marginRight: 6 }} />Findings block the goal</>}
        <span style={{ color: 'var(--text-dim)', fontWeight: 400 }}> — {r.goal}</span>
      </div>
      <div className="review-counts" style={{ display: 'flex', gap: 10, margin: '6px 0 10px' }}>
        {(['CRITICAL', 'HIGH', 'MEDIUM', 'LOW'] as Severity[]).map(s => (
          <span key={s} style={{ fontSize: 12, color: c[s] ? SEVERITY_COLOR[s] : 'var(--text-dim)' }}>
            {c[s]} {s.toLowerCase()}
          </span>
        ))}
      </div>
      {r.findings.length === 0 ? (
        <div style={{ color: 'var(--text-dim)', fontSize: 13 }}>
          No findings — the reviewer could not refute any reported value.
        </div>
      ) : (
        <ul style={{ listStyle: 'none', padding: 0, margin: 0, display: 'flex', flexDirection: 'column', gap: 8 }}>
          {r.findings.map((f: ReviewFinding) => (
            <li key={f.id}>
              <div style={{ fontSize: 12 }}>
                <span className={`sev-pill ${f.severity}`}>{f.severity}</span>
                <span style={{ color: 'var(--text)' }}> · {f.target}</span>
              </div>
              <div style={{ fontSize: 13, marginTop: 2 }}><strong>Claim:</strong> {f.claim}</div>
              <div style={{ fontSize: 13 }}><strong>Evidence:</strong> {f.evidence}</div>
              <div style={{ fontSize: 12, color: 'var(--text-dim)', marginTop: 2 }}>→ {f.suggested_action}</div>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

function SkillsPanel({ skills, loading, datasetPath, onRun, onDelete, onMarkdown }: {
  skills: SkillDef[]; loading: boolean; datasetPath: string | null;
  onRun: (name: string) => void; onDelete: (name: string) => void;
  onMarkdown: (name: string) => void;
}) {
  return (
    <div className="quick-actions" style={{ flexDirection: 'column', alignItems: 'stretch' }}>
      {skills.length === 0 ? (
        <div style={{ color: 'var(--text-dim)', fontSize: 13 }}>
          No skills captured yet. Run an analysis, then “Capture as skill”.
        </div>
      ) : (
        <ul style={{ listStyle: 'none', padding: 0, margin: 0, display: 'flex', flexDirection: 'column', gap: 6 }}>
          {skills.map(s => (
            <li key={s.name} style={{ display: 'flex', alignItems: 'center', gap: 8, flexWrap: 'wrap' }}>
              <span style={{ fontWeight: 600 }}>{s.name}</span>
              <span style={{ fontSize: 12, color: 'var(--text-dim)' }}>v{s.version}</span>
              <span style={{ fontSize: 12, fontFamily: 'var(--mono)', color: 'var(--text-dim)' }}>{s.steps.map(st => st.tool).join(' → ')}</span>
              <span style={{ flex: 1 }} />
              <button className="chip" disabled={loading || !datasetPath} onClick={() => onRun(s.name)}
                title={datasetPath ? 'Replay on the current dataset' : 'Load a dataset first'}>Replay</button>
              <button className="chip" disabled={loading} onClick={() => onMarkdown(s.name)}>SKILL.md</button>
              <button className="chip" disabled={loading} onClick={() => onDelete(s.name)}>Delete</button>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

function NcaSubjectTable({ state, limit }: { state: PharmState; limit?: number }) {
  const raw = state.nca_parameters;
  if (!raw || raw.length === 0) return null;
  const rows = bySubjectId(raw, r => r.subject);
  const shown = limit != null ? rows.slice(0, limit) : rows;
  const ss = state.nca_summary?.steady_state === true;
  const tau = rows.find(r => r.tau != null)?.tau;
  const u = ncaUnits(state);

  const summ = state.nca_summary;
  const meta = summ ? `${summ.route ?? 'extravascular'}${summ.blq ? ` · ${summ.blq.n_below_loq} BLQ` : ''}` : '';
  if (ss) {
    return (
      <div>
        <div style={{ fontSize: 12, color: 'var(--text-dim)', marginBottom: 4 }}>
          Steady-state NCA — {rows.length} subjects · AUC over τ{tau ? ` = ${tau} h` : ''}{meta && ` · ${meta}`}
        </div>
        <table className="nca-table">
          <thead>
            <tr>
              <th>ID</th><th className="num">Dose{unitSuffix(u.dose)}</th><th className="num">Cmax,ss{unitSuffix(u.conc)}</th><th className="num">Cmin{unitSuffix(u.conc)}</th>
              <th className="num">AUC<sub>τ</sub>{unitSuffix(u.auc)}</th><th className="num">Cavg{unitSuffix(u.conc)}</th>
              <th className="num">CL/F{unitSuffix(u.clf)}</th><th className="num">t½{unitSuffix(u.time)}</th><th className="num">Fluct%</th><th className="num">R<sub>ac</sub></th>
            </tr>
          </thead>
          <tbody>
            {shown.map(r => (
              <tr key={String(r.subject)}>
                <td>{r.subject}</td>
                <td className="num">{fmt(r.dose, 0)}</td>
                <td className="num">{fmt(r.Cmax)}</td>
                <td className="num">{fmt(r.Cmin ?? undefined)}</td>
                <td className="num">{fmt(r.AUC_tau ?? r.AUC_last, 1)}</td>
                <td className="num">{fmt(r.Cavg ?? undefined, 1)}</td>
                <td className="num">{fmt(r.CL_F, 2)}</td>
                <td className="num">{fmt(r.t_half, 1)}</td>
                <td className="num">{fmt(r.fluctuation_pct ?? undefined, 0)}</td>
                <td className="num">{fmt(r.accumulation_ratio ?? undefined, 2)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    );
  }

  return (
    <div>
      <div style={{ fontSize: 12, color: 'var(--text-dim)', marginBottom: 4 }}>
        Per-subject NCA — {rows.length} subjects{meta && ` · ${meta}`}
      </div>
      <table className="nca-table">
        <thead>
          <tr>
            <th>ID</th><th className="num">Dose{unitSuffix(u.dose)}</th><th className="num">Cmax{unitSuffix(u.conc)}</th><th className="num">Tmax{unitSuffix(u.time)}</th>
            <th className="num">AUC<sub>last</sub>{unitSuffix(u.auc)}</th><th className="num">AUC<sub>inf</sub>{unitSuffix(u.auc)}</th>
            <th className="num">t½{unitSuffix(u.time)}</th><th className="num">CL/F{unitSuffix(u.clf)}</th><th className="num">Vz/F{unitSuffix(u.vz)}</th><th className="num">% extrap.</th>
          </tr>
        </thead>
        <tbody>
          {shown.map(r => {
            const hi = (r.pct_AUC_extrap ?? 0) > EXTRAP_LIMIT;
            return (
              <tr key={String(r.subject)}>
                <td>{r.subject}</td>
                <td className="num">{fmt(r.dose, 0)}</td>
                <td className="num">{fmt(r.Cmax)}</td>
                <td className="num">{fmt(r.Tmax)}</td>
                <td className="num">{fmt(r.AUC_last, 1)}</td>
                <td className="num">{fmt(r.AUC_inf ?? undefined, 1)}</td>
                <td className="num">{fmt(r.t_half, 1)}</td>
                <td className="num">{fmt(r.CL_F, 2)}</td>
                <td className="num">{fmt(r.Vz_F, 1)}</td>
                <td className={`num ${hi ? 'extrap-hi' : ''}`}>
                  {fmt(r.pct_AUC_extrap ?? undefined, 1)}%{hi && <> <ExtrapBadge /></>}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

function DoseSummaryTable({ state }: { state: PharmState }) {
  const s = state.nca_summary;
  if (!s || s.by_dose.length === 0) return null;
  // Weight-based dosing yields many near-unique doses (coincidental ties aside).
  // Only show a dose-group summary when there are genuinely few dose levels with
  // replicates — i.e. CV/geomean across the group is meaningful.
  const fewLevels = s.by_dose.length <= Math.max(3, Math.floor(s.n_subjects / 3));
  const hasReplicates = s.by_dose.some(d => d.n >= 3);
  if (!fewLevels || !hasReplicates) return null;
  return (
    <div style={{ marginTop: 12 }}>
      <div style={{ fontSize: 12, color: 'var(--text-dim)', marginBottom: 4 }}>
        Dose-group summary (geometric mean · geoCV%)
      </div>
      <table className="nca-table">
        <thead>
          <tr>
            <th>Dose</th><th>N</th><th>Cmax</th><th>geoCV%</th>
            <th>AUC<sub>inf</sub></th><th>geoCV%</th><th>t½ (median)</th>
          </tr>
        </thead>
        <tbody>
          {s.by_dose.map(row => (
            <tr key={row.dose}>
              <td>{fmt(row.dose, 0)}</td>
              <td>{row.n}</td>
              <td>{fmt(row.Cmax_geomean)}</td>
              <td>{row.Cmax_geocv_pct == null ? '–' : fmt(row.Cmax_geocv_pct, 1) + '%'}</td>
              <td>{fmt(row.AUC_inf_geomean, 1)}</td>
              <td>{row.AUC_inf_geocv_pct == null ? '–' : fmt(row.AUC_inf_geocv_pct, 1) + '%'}</td>
              <td>{fmt(row.t_half_median, 1)} h</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

const NCA_PARAM_LABEL: Record<string, string> = {
  Cmax: 'Cmax', Tmax: 'Tmax', Cmin: 'Cmin', Ctrough: 'Ctrough', Cavg: 'Cavg', Clast: 'Clast', Tlast: 'Tlast',
  AUC_last: 'AUClast', AUC_inf: 'AUCinf', AUC_tau: 'AUCτ', pct_AUC_extrap: '%AUC extrap',
  lambda_z: 'λz', t_half: 't½', MRT: 'MRT', CL_F: 'CL/F', Vz_F: 'Vz/F', Vss: 'Vss',
  accumulation_ratio: 'Rac', fluctuation_pct: 'Fluctuation %', swing_pct: 'Swing %',
};

/** Standard NCA summary statistics per parameter (N, mean, SD, CV%, median,
 *  min, max, geometric mean, geo-CV%), overall or per dose group. */
function DescriptiveStatsTable({ state }: { state: PharmState }) {
  const groups = state.nca_summary?.descriptive ?? [];
  const [gi, setGi] = useState(0);
  if (!groups.length) return null;
  const g = groups[Math.min(gi, groups.length - 1)];
  const d = (v: number | null, digits = 3) => (v == null ? '–' : fmt(v, digits));
  const pct = (v: number | null) => (v == null ? '–' : `${fmt(v, 1)}%`);
  return (
    <div style={{ marginTop: 12 }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 4 }}>
        <span style={{ fontSize: 12, color: 'var(--text-dim)' }}>Summary statistics</span>
        {groups.length > 1 && (
          <select value={gi} onChange={e => setGi(Number(e.target.value))} aria-label="Summary statistics group"
            style={{ fontSize: 12, padding: '2px 6px', borderRadius: 6, border: '1px solid var(--border-strong)', background: 'var(--bg-card)', color: 'var(--text-h)' }}>
            {groups.map((x, i) => <option key={String(x.group)} value={i}>{x.label} (n={x.n})</option>)}
          </select>
        )}
        {groups.length === 1 && <span style={{ fontSize: 11, color: 'var(--text-dim)' }}>· {g.label} (n={g.n})</span>}
      </div>
      <div style={{ overflowX: 'auto' }}>
        <table className="nca-table">
          <thead>
            <tr><th>Parameter</th><th>N</th><th>Mean</th><th>SD</th><th>CV%</th><th>Median</th><th>Min</th><th>Max</th><th>Geo. mean</th><th>Geo. CV%</th></tr>
          </thead>
          <tbody>
            {g.parameters.map(p => (
              <tr key={p.parameter}>
                <td>{NCA_PARAM_LABEL[p.parameter] ?? p.parameter}</td>
                <td>{p.n}</td>
                <td>{d(p.mean)}</td><td>{d(p.sd)}</td><td>{pct(p.cv_pct)}</td>
                <td>{d(p.median)}</td><td>{d(p.min)}</td><td>{d(p.max)}</td>
                <td>{d(p.geomean)}</td><td>{pct(p.geocv_pct)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div style={{ fontSize: 11, color: 'var(--text-dim)', marginTop: 4 }}>
        SD uses n−1; CV% = SD/mean; geometric CV% = √(exp(s²<sub>log</sub>) − 1); N counts subjects with a valid value.
      </div>
    </div>
  );
}

/** One-line geometric-mean summary under the NCA table, built ONLY from the
 *  fields present in nca_summary.descriptive (group 'all' or the first
 *  group). Units are never invented: each comes from ncaUnits (the dataset's
 *  own column labels). Returns null when nothing can be stated. */
function ncaFooterParts(state: PharmState): React.ReactNode[] {
  const groups = state.nca_summary?.descriptive ?? [];
  const g = groups.find(x => x.group === 'all') ?? groups[0];
  if (!g) return [];
  const param = (name: string) => g.parameters.find(p => p.parameter === name);
  const u = ncaUnits(state);
  const parts: React.ReactNode[] = [];
  const cl = param('CL_F');
  if (cl?.geomean != null) {
    parts.push(<span key="cl">Geometric mean CL/F <b>{fmt(cl.geomean, 3)}{u.clf ? ` ${u.clf}` : ''}</b>{cl.geocv_pct != null && <> (gCV {fmt(cl.geocv_pct, 1)} %)</>}</span>);
  }
  const th = param('t_half');
  if (th?.median != null) {
    parts.push(<span key="th">t½ <b>{fmt(th.median, 1)}{u.time ? ` ${u.time}` : ''}</b></span>);
  }
  const auc = param('AUC_inf');
  if (auc?.geomean != null) {
    parts.push(<span key="auc">AUCinf <b>{fmt(auc.geomean, 1)}{u.auc ? ` ${u.auc}` : ''}</b>{auc.geocv_pct != null && <> (gCV {fmt(auc.geocv_pct, 1)} %)</>}</span>);
  }
  return parts;
}

function NcaTable({ state }: { state: PharmState }) {
  const n = state.nca_parameters?.length ?? 0;
  const collapsible = n > NCA_ROWS_THRESHOLD;
  const [showAll, setShowAll] = useState(false);
  const [showStats, setShowStats] = useState(false);
  const hasStats = !!state.nca_summary?.descriptive?.length || !!state.nca_summary?.by_dose?.length;
  const footer = ncaFooterParts(state);
  return (
    <div>
      <NcaSubjectTable state={state} limit={collapsible && !showAll ? NCA_ROWS_COLLAPSED : undefined} />
      {(footer.length > 0 || collapsible || hasStats) && (
        <div className="nca-foot">
          <span className="nca-foot-stats">
            {footer.map((f, i) => <span key={i}>{i > 0 && ' · '}{f}</span>)}
          </span>
          <span className="nca-foot-links">
            {collapsible && (
              <button type="button" className="link-btn" onClick={() => setShowAll(v => !v)}>
                {showAll ? 'Show fewer' : `Show all ${n} subjects`}
              </button>
            )}
            {hasStats && (
              <button type="button" className="link-btn" aria-expanded={showStats} onClick={() => setShowStats(v => !v)}>
                Summary statistics
              </button>
            )}
          </span>
        </div>
      )}
      {showStats && (
        <>
          <DescriptiveStatsTable state={state} />
          <DoseSummaryTable state={state} />
        </>
      )}
    </div>
  );
}

const PROVIDER_LABEL: Record<LlmProvider, string> = {
  mock: 'Mock (keyless, deterministic)', local: 'Local — Ollama (free)',
  openai: 'ChatGPT — OpenAI API key', anthropic: 'Claude — Anthropic API key',
};

/** Header popover: pick the model that routes requests and picks tools.
 *  Keys are sent to the local backend and held in its memory only. */
function LlmSettings({ onApplied, onClose }: { onApplied: (label: string) => void; onClose: () => void }) {
  const [cfg, setCfg] = useState<LlmConfig | null>(null);
  const [provider, setProvider] = useState<LlmProvider>('mock');
  const [model, setModel] = useState('');
  const [apiKey, setApiKey] = useState('');
  const [baseUrl, setBaseUrl] = useState('');
  const [busy, setBusy] = useState<'test' | 'apply' | null>(null);
  const [note, setNote] = useState<{ ok: boolean; text: string } | null>(null);

  useEffect(() => {
    api.getLlm().then(c => {
      setCfg(c); setProvider(c.current.provider); setModel(c.current.model);
      setBaseUrl(c.current.provider === 'local' && c.current.base_url ? c.current.base_url : '');
    }).catch(e => setNote({ ok: false, text: (e as Error).message }));
  }, []);

  const pick = (p: LlmProvider) => {
    setProvider(p); setNote(null); setApiKey('');
    setModel(cfg && cfg.current.provider === p ? cfg.current.model : (cfg?.defaults[p] ?? ''));
  };
  const needsKey = provider === 'openai' || provider === 'anthropic';
  const keyHeld = !!cfg && cfg.current.provider === provider && cfg.current.has_key;

  const submit = async (testOnly: boolean) => {
    setBusy(testOnly ? 'test' : 'apply'); setNote(null);
    try {
      const r = await api.setLlm({ provider, model, ...(apiKey ? { api_key: apiKey } : {}),
        ...(baseUrl ? { base_url: baseUrl } : {}), test_only: testOnly });
      setNote({ ok: true, text: testOnly ? `Test OK — ${r.detail}` : `Switched — ${r.detail}` });
      if (!testOnly) { setApiKey(''); const c = await api.getLlm(); setCfg(c); onApplied(c.label); }
    } catch (e) {
      setNote({ ok: false, text: errorText(e) });
    } finally { setBusy(null); }
  };

  const field = { width: '100%', fontSize: 12, padding: '5px 8px', borderRadius: 6,
    border: '1px solid var(--border)', background: 'var(--bg)', color: 'var(--text)' } as const;
  return (
    <div style={{ width: 360, padding: 2, background: 'var(--bg-card)', textAlign: 'left' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8 }}>
        <strong style={{ fontSize: 13 }}>Language model</strong>
        <button className="btn btn-ghost" style={{ padding: '2px 8px', fontSize: 11 }} onClick={onClose}>close</button>
      </div>
      <div style={{ fontSize: 11, color: 'var(--text-dim)', marginBottom: 8 }}>
        Only routing and tool choice use the model; every number comes from deterministic code.
        API keys stay in the local backend&apos;s memory (never on disk, never shown again).
      </div>
      {(cfg?.providers ?? (['mock', 'local', 'openai', 'anthropic'] as LlmProvider[])).map(p => (
        <label key={p} style={{ display: 'flex', gap: 8, alignItems: 'center', fontSize: 12, padding: '3px 0' }}>
          <input type="radio" name="llm-provider" checked={provider === p} onChange={() => pick(p)} />
          {PROVIDER_LABEL[p]}
          {cfg?.current.provider === p && <span style={{ color: 'var(--green)', fontSize: 11 }}>· active</span>}
        </label>
      ))}
      {provider !== 'mock' && (
        <div style={{ marginTop: 8 }}>
          <div style={{ fontSize: 11, color: 'var(--text-dim)', marginBottom: 2 }}>Model</div>
          <input style={field} value={model} onChange={e => setModel(e.target.value)} list="llm-local-models"
            placeholder={cfg?.defaults[provider]} />
          {provider === 'local' && (
            <datalist id="llm-local-models">{(cfg?.local_models ?? []).map(m => <option key={m} value={m} />)}</datalist>
          )}
          {provider === 'local' && !(cfg?.local_models ?? []).length && (
            <div style={{ fontSize: 11, color: 'var(--yellow)', marginTop: 3 }}>
              No Ollama models found — run <code>ollama pull gemma4:31b</code> (Ollama must be running).
            </div>
          )}
        </div>
      )}
      {needsKey && (
        <div style={{ marginTop: 8 }}>
          <div style={{ fontSize: 11, color: 'var(--text-dim)', marginBottom: 2 }}>
            API key {keyHeld ? '(one is held — leave blank to keep it)' : ''}
          </div>
          <input style={field} type="password" value={apiKey} onChange={e => setApiKey(e.target.value)}
            placeholder={provider === 'openai' ? 'sk-…' : 'sk-ant-…'} autoComplete="off" />
        </div>
      )}
      {(provider === 'local' || provider === 'openai') && (
        <div style={{ marginTop: 8 }}>
          <div style={{ fontSize: 11, color: 'var(--text-dim)', marginBottom: 2 }}>Server URL (optional)</div>
          <input style={field} value={baseUrl} onChange={e => setBaseUrl(e.target.value)}
            placeholder={provider === 'local' ? 'http://127.0.0.1:11434/v1 (Ollama) · LM Studio: http://127.0.0.1:1234/v1' : 'https://api.openai.com/v1'} />
        </div>
      )}
      <div style={{ display: 'flex', gap: 6, marginTop: 10 }}>
        <button className="btn btn-ghost" disabled={busy !== null} onClick={() => submit(true)}>
          {busy === 'test' ? 'Testing…' : 'Test'}
        </button>
        <button className="btn btn-green" disabled={busy !== null} onClick={() => submit(false)}>
          {busy === 'apply' ? 'Switching…' : 'Use this model'}
        </button>
      </div>
      {note && (
        <div style={{ fontSize: 11, marginTop: 8, color: note.ok ? 'var(--green)' : 'var(--red)', wordBreak: 'break-word' }}>
          {note.text}
        </div>
      )}
    </div>
  );
}

/** Sentence-case axis titles for the concentration–time chart. The unit is
 *  shown only when the raw column label states it — never invented. */
function spaghettiAxisLabels(data: SpaghettiData): { x: string; y: string } {
  const xUnit = labelUnit(data.x_label);
  const hours = /hour|\(h\)|\bh\b/i.test(data.x_label);
  const yUnit = labelUnit(data.y_label);
  return {
    x: `Time${hours ? ' (h)' : xUnit ? ` (${xUnit})` : ''}`,
    y: `Concentration${yUnit ? ` (${yUnit})` : ''}`,
  };
}

function SpaghettiChart({ data, flagged = [] }: { data: SpaghettiData; flagged?: string[] }) {
  const [logY, setLogY] = useState(data.log_scale);
  const [individual, setIndividual] = useState(false);
  // user-chosen time window (null = the data's full range); applies to both views
  const [xr, setXr] = useState<[number, number] | null>(null);
  const [expanded, setExpanded] = useState(false);
  const flaggedSet = new Set(flagged);
  const axis = spaghettiAxisLabels(data);

  const W = 760, H = 220, ml = 52, mr = 12, mt = 12, mb = 38;
  const allYAll = data.series.flatMap(s => s.y).filter(v => v > 0);
  const allXAll = data.series.flatMap(s => s.x).filter(isFinite);
  if (!allYAll.length || !allXAll.length) return <div style={{ fontSize: 12, color: 'var(--text-dim)' }}>No data to plot.</div>;

  // x-range from the DATA, not from 0: a steady-state window sampled at
  // 168–192 h must not be squashed into the last 12% of a 0–192 h axis.
  const dataLo = Math.min(...allXAll), dataHi = Math.max(...allXAll);
  const xlo = xr ? Math.max(dataLo, Math.min(xr[0], xr[1])) : dataLo;
  const xhi = xr ? Math.min(dataHi, Math.max(xr[0], xr[1])) : dataHi;
  const inRange = (x: number) => x >= xlo - 1e-9 && x <= xhi + 1e-9;
  const allY = data.series.flatMap(s => s.y.filter((v, j) => v > 0 && inRange(s.x[j])));
  if (!allY.length) return <div>{/* nothing inside the window */}
    <div style={{ fontSize: 12, color: 'var(--text-dim)' }}>No observations in {fmt(xlo, 1)}–{fmt(xhi, 1)}.</div>
    <button type="button" className="link-btn" style={{ marginTop: 4 }} onClick={() => setXr(null)}>Reset time window</button>
  </div>;
  const xpad = (xhi - xlo || xhi || 1) * 0.04;
  const xmin = xlo - xpad < 0 && xlo >= 0 ? 0 : xlo - xpad;
  const xmax = xhi + xpad;
  const ymin = Math.min(...allY), ymax = Math.max(...allY) * 1.1;
  const cw = W - ml - mr, ch = H - mt - mb;
  const sx = (x: number) => ml + ((x - xmin) / (xmax - xmin || 1)) * cw;
  const lmin10 = Math.log10(ymin * 0.8), lmax10 = Math.log10(ymax);
  const syLog = (y: number) => y > 0 ? H - mb - (Math.log10(y) - lmin10) / (lmax10 - lmin10) * ch : H - mb;
  const syLin = (y: number) => H - mb - (y / ymax) * ch;
  const sy = logY ? syLog : syLin;

  // Series colour: with a flagged subject present, the flagged one is red and
  // the rest recede to the accent (as in the comp); otherwise the per-subject
  // palette so individual curves can still be told apart.
  const anyFlagged = data.series.some(s => flaggedSet.has(String(s.id)));
  const seriesColor = (id: string, i: number) =>
    flaggedSet.has(id) ? 'var(--red)' : anyFlagged ? 'var(--accent)' : SPAG_PALETTE[i % SPAG_PALETTE.length];
  const seriesOpacity = (id: string) => (flaggedSet.has(id) ? 0.9 : anyFlagged ? 0.5 : 0.7);

  const caption = `${data.n_subjects} subjects`
    + (data.blq_excluded > 0 ? ` · ${data.blq_excluded} points below the LLOQ excluded` : '')
    + flaggedSentence(flagged.filter(id => data.series.some(s => String(s.id) === id)));

  const controls = (
    <div className="chart-controls">
      <div className="seg" role="group" aria-label="Y axis scale">
        <button type="button" aria-pressed={logY} onClick={() => setLogY(true)}>Log</button>
        <button type="button" aria-pressed={!logY} onClick={() => setLogY(false)}>Linear</button>
      </div>
      <div className="seg" role="group" aria-label="View">
        <button type="button" aria-pressed={!individual} onClick={() => setIndividual(false)}>Overlay</button>
        <button type="button" aria-pressed={individual} onClick={() => setIndividual(true)}>Individual</button>
      </div>
      <XRangeSlider min={dataLo} max={dataHi} value={[xlo, xhi]} label="Time window"
        onChange={v => setXr(v)} onReset={() => setXr(null)} />
      {!individual && (
        <button type="button" className="chart-expand" aria-label="Open concentration–time chart full screen"
          title="Open full screen — wheel to zoom, drag to pan" onClick={() => setExpanded(true)}>
          <Maximize2 size={13} aria-hidden="true" />
        </button>
      )}
    </div>
  );

  if (individual) {
    const sw = 210, sh = 150, sml = 42, smr = 8, smt = 8, smb = 28;
    const xTicksI = niceTicks(xmin, xmax, 3).filter(v => v >= xmin && v <= xmax);
    const scx = (x: number) => sml + ((x - xmin) / (xmax - xmin || 1)) * (sw - sml - smr);
    const tickLabel = (v: number) => v >= 1000 ? `${(v / 1000).toFixed(0)}k` : v >= 1 ? String(Math.round(v)) : v.toPrecision(1);
    return (
      <figure className="chart-figure">
        {controls}
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6 }}>
          {data.series.map((s, i) => {
            const id = String(s.id);
            const color = seriesColor(id, i);
            const pts = s.x.map((x, j) => ({ x, y: s.y[j] })).filter(p => p.y > 0 && inRange(p.x));
            if (!pts.length) return null;
            const ays = pts.map(p => p.y);
            const yTop = Math.max(...ays) * 1.1, yBot = Math.min(...ays) * 0.8;
            const aln = Math.log10(yBot), alx = Math.log10(yTop);
            const ach = sh - smt - smb;
            const scy = (y: number) => logY
              ? (y > 0 ? sh - smb - (Math.log10(y) - aln) / (alx - aln || 1) * ach : sh - smb)
              : sh - smb - (y / yTop) * ach;
            // y ticks: decades inside the panel's range (log) or quarter steps (linear);
            // fall back to the panel's min/max so every panel shows at least two labels
            let yTicksI: number[] = logY
              ? Array.from({ length: Math.ceil(alx) - Math.floor(aln) + 1 }, (_, k) => Math.pow(10, Math.floor(aln) + k))
                .filter(v => v >= yBot && v <= yTop)
              : [0.5, 1.0].map(f => f * yTop / 1.1);
            if (yTicksI.length < 2) yTicksI = [Math.min(...ays), Math.max(...ays)];
            const poly = pts.map(p => `${scx(p.x).toFixed(1)},${scy(p.y).toFixed(1)}`).join(' ');
            return (
              <Zoomable key={s.id} title={`${s.id} — ${axis.y} vs ${axis.x}`}>
              <svg viewBox={`0 0 ${sw} ${sh}`} width={sw} role="img"
                aria-label={`Subject ${s.id}: ${axis.y} vs ${axis.x}${flaggedSet.has(id) ? ', terminal phase flagged' : ''}`}
                style={{ background: 'rgba(31,102,166,0.03)', borderRadius: 4, border: `1px solid ${flaggedSet.has(id) ? 'var(--red)' : 'var(--border)'}`, display: 'block' }}>
                <line x1={sml} y1={sh - smb} x2={sw - smr} y2={sh - smb} stroke="var(--border)" />
                <line x1={sml} y1={smt} x2={sml} y2={sh - smb} stroke="var(--border)" />
                {yTicksI.map((v, k) => (
                  <g key={k}>
                    <line x1={sml - 3} y1={scy(v)} x2={sml} y2={scy(v)} stroke="var(--text-dim)" />
                    <text x={sml - 4} y={scy(v) + 3} textAnchor="end" fontSize="9" {...TICK_TEXT}>{tickLabel(v)}</text>
                  </g>
                ))}
                {xTicksI.map((v, k) => (
                  <g key={k}>
                    <line x1={scx(v)} y1={sh - smb} x2={scx(v)} y2={sh - smb + 3} stroke="var(--text-dim)" />
                    <text x={scx(v)} y={sh - smb + 12} textAnchor="middle" fontSize="9" {...TICK_TEXT}>{v}</text>
                  </g>
                ))}
                {pts.length > 1 && <polyline points={poly} fill="none" stroke={color} strokeWidth="1.3" strokeOpacity="0.8" />}
                {pts.map((p, j) => <circle key={j} cx={scx(p.x)} cy={scy(p.y)} r="2" fill={color} />)}
                <text x={(sml + sw - smr) / 2} y={sh - 3} textAnchor="middle" fontSize="10" fontWeight="600" fill="var(--text)">{s.id}</text>
                <text x={10} y={(smt + sh - smb) / 2} textAnchor="middle" fontSize="9" fill="var(--text-dim)"
                  transform={`rotate(-90 10 ${(smt + sh - smb) / 2})`}>{logY ? 'log' : 'linear'}</text>
              </svg>
              </Zoomable>
            );
          })}
        </div>
        <div style={{ fontSize: 11, color: 'var(--text)', marginTop: 6 }}>
          x: <b>{axis.x}</b> {fmt(xlo, 1)}–{fmt(xhi, 1)} (shared) · y: <b>{axis.y}</b>{logY ? ' (log scale)' : ''}, scaled per subject
        </div>
        <figcaption className="figcap">{caption}</figcaption>
      </figure>
    );
  }

  // Y axis ticks
  const yTicks = logY
    ? (() => {
        const lo = Math.floor(lmin10), hi = Math.ceil(lmax10);
        return Array.from({ length: hi - lo + 1 }, (_, k) => Math.pow(10, lo + k))
          .filter(v => { const yy = syLog(v); return yy >= mt && yy <= H - mb; });
      })()
    : [0.25, 0.5, 0.75, 1.0].map(f => f * ymax);

  const xTicks = niceTicks(xmin, xmax, 6).filter(v => v >= xmin && v <= xmax);

  // Flagged series are drawn last so they sit on top of the others.
  const ordered = data.series.map((s, i) => ({ s, i }))
    .sort((a, b) => Number(flaggedSet.has(String(a.s.id))) - Number(flaggedSet.has(String(b.s.id))));

  const overlaySvg = (
      <svg viewBox={`0 0 ${W} ${H}`} style={{ width: '100%', display: 'block' }} role="img"
        aria-label={`${logY ? 'Log-scale' : 'Linear'} ${axis.y} vs ${axis.x} for ${data.n_subjects} subjects`}>
        <line x1={ml} y1={H - mb} x2={W - mr} y2={H - mb} stroke="var(--border)" />
        <line x1={ml} y1={mt} x2={ml} y2={H - mb} stroke="var(--border)" />
        {yTicks.map((v, k) => {
          const yy = sy(v);
          const lbl = v >= 1000 ? `${(v / 1000).toFixed(0)}k` : v >= 1 ? String(Math.round(v)) : v.toPrecision(1);
          return (
            <g key={k}>
              <line x1={ml - 3} y1={yy} x2={ml} y2={yy} stroke="var(--text-dim)" />
              <text x={ml - 5} y={yy + 3.5} textAnchor="end" fontSize="10" {...TICK_TEXT}>{lbl}</text>
            </g>
          );
        })}
        {xTicks.map((v, k) => (
          <g key={k}>
            <line x1={sx(v)} y1={H - mb} x2={sx(v)} y2={H - mb + 3} stroke="var(--text-dim)" />
            <text x={sx(v)} y={H - mb + 13} textAnchor="middle" fontSize="10" {...TICK_TEXT}>{v}</text>
          </g>
        ))}
        <text x={(ml + W - mr) / 2} y={H - 6} textAnchor="middle" {...AXIS_TITLE}>{axis.x}</text>
        <text x={14} y={(mt + H - mb) / 2} textAnchor="middle" {...AXIS_TITLE}
          transform={`rotate(-90 14 ${(mt + H - mb) / 2})`}>{axis.y}</text>
        {ordered.map(({ s, i }) => {
          const id = String(s.id);
          const isFlagged = flaggedSet.has(id);
          const color = seriesColor(id, i);
          const pts = s.x.map((x, j) => ({ x, y: s.y[j] })).filter(p => p.y > 0 && inRange(p.x));
          if (!pts.length) return null;
          const poly = pts.map(p => `${sx(p.x).toFixed(1)},${sy(p.y).toFixed(1)}`).join(' ');
          const last = pts[pts.length - 1];
          return (
            <g key={s.id}>
              {pts.length > 1 && <polyline points={poly} fill="none" stroke={color}
                strokeWidth={isFlagged ? '1.5' : '1.2'} strokeOpacity={seriesOpacity(id)} />}
              {pts.map((p, j) => <circle key={j} cx={sx(p.x)} cy={sy(p.y)} r={isFlagged ? 2.5 : 2}
                fill={color} fillOpacity={isFlagged ? 0.9 : 0.85} />)}
              {isFlagged && (
                <text x={Math.min(sx(last.x), W - mr) - 4} y={sy(last.y) - 5} textAnchor="end"
                  fontSize="11" fill="var(--red)">subject {id}</text>
              )}
            </g>
          );
        })}
      </svg>
  );

  return (
    <figure className="chart-figure">
      {controls}
      {overlaySvg}
      {expanded && <ZoomModal title="Concentration–time" onClose={() => setExpanded(false)}>{overlaySvg}</ZoomModal>}
      <figcaption className="figcap">{caption}</figcaption>
    </figure>
  );
}

interface LzManualFit {
  lambda_z: number; lambda_z_intercept: number;
  t_half: number; r2_adj: number; n_pts: number;
  lz_x: number[]; lz_y: number[];
  fit_x: number[]; fit_y: number[];
}

/** Panels shown before "All N subjects". */
const LZ_PANELS_COLLAPSED = 4;
/** Radius of the transparent hit circle around each λz point: 12 viewBox
 *  units = 24px at the panel's 1:1 rendering, the minimum touch target. Per
 *  panel it is clamped to half the smallest x-gap between neighbouring points
 *  (never below LZ_HIT_R_MIN) so hit circles cannot cover a neighbour's dot —
 *  the later sibling would otherwise win the click and toggle the wrong point. */
const LZ_HIT_R = 12;
const LZ_HIT_R_MIN = 4;

function NcaLzPlot({ data, sessionId, flagged = {}, hours = false }: {
  data: NcaPlotData; sessionId: string;
  /** subject id → %AUC extrapolated, for subjects over the 20 % limit (from the NCA table). */
  flagged?: Record<string, number>;
  /** true when the dataset's time axis is labelled in hours — the only case an 'h' is printed. */
  hours?: boolean;
}) {
  const subjects = useMemo(() => bySubjectId(data.subjects, s => s.id), [data.subjects]);

  const [selections, setSelections] = useState<Record<string, Set<string>>>(() => {
    const s: Record<string, Set<string>> = {};
    for (const sub of subjects) s[sub.id] = new Set(sub.lz_x.map(v => v.toFixed(4)));
    return s;
  });
  const [localFits, setLocalFits] = useState<Record<string, LzManualFit>>({});
  const [loadingId, setLoadingId] = useState<string | null>(null);
  const [errors, setErrors] = useState<Record<string, string>>({});
  const [showAll, setShowAll] = useState(false);

  const toggle = (sid: string, tk: string) => {
    setSelections(prev => {
      const next = new Set(prev[sid]);
      if (next.has(tk)) next.delete(tk); else next.add(tk);
      return { ...prev, [sid]: next };
    });
    setErrors(prev => ({ ...prev, [sid]: '' }));
  };

  const doRefit = async (s: LzSubject) => {
    const sel = selections[s.id] ?? new Set<string>();
    const pts = s.x.map((x, i) => ({ x, y: s.y[i] }))
      .filter(p => p.y > 0 && sel.has(p.x.toFixed(4)));
    if (pts.length < 3) { setErrors(prev => ({ ...prev, [s.id]: 'Select ≥ 3 points' })); return; }
    setLoadingId(s.id);
    setErrors(prev => ({ ...prev, [s.id]: '' }));
    try {
      const res = await api.refitLz(sessionId, {
        subject: s.id,
        selected_times: pts.map(p => p.x),
        selected_concs: pts.map(p => p.y),
      });
      setLocalFits(prev => ({ ...prev, [s.id]: res }));
    } catch (e) {
      setErrors(prev => ({ ...prev, [s.id]: (e as Error).message }));
    } finally {
      setLoadingId(null);
    }
  };

  const reset = (s: LzSubject) => {
    setSelections(prev => ({ ...prev, [s.id]: new Set(s.lz_x.map(v => v.toFixed(4))) }));
    setLocalFits(prev => { const n = { ...prev }; delete n[s.id]; return n; });
    setErrors(prev => ({ ...prev, [s.id]: '' }));
  };

  if (!subjects.length) return null;
  const sw = 164, sh = 84, sml = 34, smr = 6, smt = 10, smb = 20;
  const h = hours ? ' h' : '';
  const collapsible = subjects.length > LZ_PANELS_COLLAPSED;
  const visible = collapsible && !showAll ? subjects.slice(0, LZ_PANELS_COLLAPSED) : subjects;
  // One sentence per flagged subject, from the NCA row (n_pts = points in the
  // current λz fit; the %AUC figure is the NCA table's own value).
  const flaggedSentences = subjects
    .filter(s => flagged[s.id] != null)
    .map(s => ` Subject ${s.id} uses ${s.n_pts ?? '?'} late points and extrapolates ${fmt(flagged[s.id], 1)} % of its AUC.`)
    .join('');

  return (
    <div className="lz-block">
      <div className="lz-head">
        <p className="lz-intro">Click points to include or exclude them, then refit.{flaggedSentences}</p>
        {collapsible && (
          <button type="button" className="link-btn" aria-expanded={showAll} onClick={() => setShowAll(v => !v)}>
            {showAll ? `First ${LZ_PANELS_COLLAPSED} subjects` : `All ${subjects.length} subjects`}
          </button>
        )}
      </div>
      <div className="lz-grid">
        {visible.map((s: LzSubject) => {
          const sel = selections[s.id] ?? new Set<string>();
          const fit = localFits[s.id];
          const isLoading = loadingId === s.id;
          const isFlagged = flagged[s.id] != null;

          const allObs = s.x.map((x, i) => ({ x, y: s.y[i] })).filter(p => p.y > 0);
          const activeFitX = fit ? fit.fit_x : s.fit_x;
          const activeFitY = fit ? fit.fit_y : s.fit_y;
          const activeTHalf = fit ? fit.t_half : s.t_half;
          const activeR2 = fit ? fit.r2_adj : s.r2_adj;
          const activeN = fit ? fit.n_pts : s.n_pts;

          const allY = [...allObs.map(p => p.y), ...activeFitY].filter(v => v > 0);
          const allX = [...allObs.map(p => p.x), ...activeFitX].filter(isFinite);
          if (!allY.length || !allX.length) return null;

          const xmax = Math.max(...allX) * 1.05;
          const lmin = Math.log10(Math.min(...allY) * 0.75);
          const lmax = Math.log10(Math.max(...allY) * 1.3);
          const ach = sh - smt - smb;
          const scx = (x: number) => sml + (x / xmax) * (sw - sml - smr);
          const scy = (y: number) => y > 0 ? sh - smb - (Math.log10(y) - lmin) / (lmax - lmin) * ach : sh - smb;

          const ylo = Math.floor(lmin), yhi = Math.ceil(lmax);
          const yTicks = Array.from({ length: yhi - ylo + 1 }, (_, k) => Math.pow(10, ylo + k))
            .filter(v => { const yy = scy(v); return yy >= smt && yy <= sh - smb; });

          const fitPoly = activeFitX.map((x, i) =>
            `${scx(x).toFixed(1)},${scy(activeFitY[i]).toFixed(1)}`).join(' ');

          // Hit radius for this panel: no circle may reach a neighbour's centre.
          const obsPx = allObs.map(p => scx(p.x)).sort((a, b) => a - b);
          const minGap = obsPx.slice(1).reduce((m, x, k) => Math.min(m, x - obsPx[k]), Infinity);
          const hitR = Math.max(LZ_HIT_R_MIN, Math.min(LZ_HIT_R, minGap / 2));

          const nSel = sel.size;
          const canRefit = nSel >= 3;
          const tHalfText = activeTHalf != null ? `t½ ${activeTHalf.toFixed(1)}${h}` : null;
          const svgLabel = `Subject ${s.id} terminal slope, ${activeN ?? '?'} points`
            + (activeTHalf != null ? `, half-life ${activeTHalf.toFixed(1)}${hours ? ' hours' : ''}` : '')
            + (isFlagged ? `, extrapolation ${fmt(flagged[s.id], 1)} percent, over the limit` : '');

          return (
            <figure key={s.id} className={`lz-panel ${isFlagged ? 'flagged' : ''}`}>
              <svg viewBox={`0 0 ${sw} ${sh}`} width={sw} role="group" aria-label={svgLabel} className="lz-svg">
                <line x1={sml} y1={sh - smb} x2={sw - smr} y2={sh - smb} stroke="var(--border)" />
                <line x1={sml} y1={smt} x2={sml} y2={sh - smb} stroke="var(--border)" />
                {yTicks.map((v, k) => {
                  const yy = scy(v);
                  return (
                    <g key={k}>
                      <line x1={sml - 3} y1={yy} x2={sml} y2={yy} stroke="var(--border)" />
                      <text x={sml - 4} y={yy + 3} textAnchor="end" fontSize="9" {...TICK_TEXT}>
                        {v >= 1000 ? `${(v / 1000).toFixed(0)}k` : v >= 1 ? String(Math.round(v)) : v.toPrecision(1)}
                      </text>
                    </g>
                  );
                })}
                {[0.5, 1.0].map((f, k) => {
                  const v = Math.round(f * xmax);
                  return v > 0 ? (
                    <g key={k}>
                      <line x1={scx(v)} y1={sh - smb} x2={scx(v)} y2={sh - smb + 3} stroke="var(--border)" />
                      <text x={scx(v)} y={sh - smb + 12} textAnchor="middle" fontSize="9" {...TICK_TEXT}>{v}</text>
                    </g>
                  ) : null;
                })}
                {activeFitX.length > 1 && (
                  <polyline points={fitPoly} fill="none" stroke={isFlagged ? 'var(--yellow)' : 'var(--text-dim)'}
                    strokeWidth="1.4" strokeDasharray="4 3" />
                )}
                {allObs.map((p, j) => {
                  const tk = p.x.toFixed(4);
                  const isSelected = sel.has(tk);
                  const cx = scx(p.x), cy = scy(p.y);
                  return (
                    <g key={j}>
                      {/* transparent hit target (≤24px, clamped so neighbours never overlap), keyboard-reachable; the visible dot sits on top */}
                      <circle className="lz-hit" cx={cx} cy={cy} r={hitR} fill="transparent"
                        tabIndex={0} role="button" aria-pressed={isSelected}
                        aria-label={`t = ${p.x}: ${isSelected ? 'included in λz' : 'excluded from λz'}`}
                        onClick={() => toggle(s.id, tk)}
                        onKeyDown={e => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); toggle(s.id, tk); } }} />
                      <circle cx={cx} cy={cy} r={isSelected ? 3 : 2.5} pointerEvents="none"
                        fill={isSelected ? 'var(--accent)' : 'var(--border-strong)'} />
                    </g>
                  );
                })}
              </svg>
              <figcaption className="lz-cap">
                <span className="lz-cap-row lz-cap-main">
                  <b>{s.id}</b>{tHalfText && <> · {tHalfText}</>} · {activeN ?? '?'} pts
                </span>
                <span className="lz-cap-row">
                  <span>
                    {activeR2 != null ? <>R² {activeR2.toFixed(3)}</> : 'R² –'}
                    {nSel !== activeN && <> · {nSel} selected</>}
                    {fit && <> · manual</>}
                  </span>
                  <span className="lz-links">
                    <button type="button" className="link-btn" onClick={() => doRefit(s)} disabled={!canRefit || isLoading}
                      title={canRefit ? 'Refit λz with the selected points' : 'Select at least 3 points'}>
                      {isLoading ? 'Refitting…' : 'Refit'}
                    </button>
                    <button type="button" className="link-btn" onClick={() => reset(s)} disabled={isLoading}>Reset</button>
                  </span>
                </span>
              </figcaption>
              {errors[s.id] && <div className="lz-error" role="alert">{errors[s.id]}</div>}
            </figure>
          );
        })}
      </div>
      {/* swatches are empty coloured spans, not glyphs, so no low-contrast "text" is rendered */}
      <div className="lz-legend">
        <span><span className="lz-swatch dot" style={{ background: 'var(--accent)' }} aria-hidden="true" /> Included in λz</span>
        <span><span className="lz-swatch dot" style={{ background: 'var(--border-strong)' }} aria-hidden="true" /> Excluded</span>
        <span><span className="lz-swatch dash" aria-hidden="true" /> Regression fit (amber when flagged)</span>
      </div>
    </div>
  );
}


const ROLE_OPTIONS = ['', 'ID', 'TIME', 'TAD', 'DV', 'AMT', 'EVID', 'MDV', 'CMT',
  'II', 'ADDL', 'DVID', 'CENS', 'ROUTE', 'PD'];

type ColMeta = { name: string; dtype: string; role: string };

function RolesEditor({ state, onApply, loading }:
  { state: PharmState; onApply: (o: Record<string, string>) => void; loading: boolean }) {
  const meta = state.dataset_metadata as { columns?: ColMeta[]; detected_roles?: Record<string, string> } | null;
  const cols = meta?.columns ?? [];
  const roles = meta?.detected_roles ?? {};
  const [edits, setEdits] = useState<Record<string, string>>({});
  if (cols.length === 0) return null;
  const roleFor = (c: string) => edits[c] ?? roles[c] ?? '';
  return (
    <div style={{ maxWidth: 560 }}>
      <div style={{ fontSize: 12, color: 'var(--text-dim)', marginBottom: 6 }}>
        Column roles — override any auto-detected mapping, then apply.
      </div>
      <table className="nca-table">
        <thead><tr><th>Column</th><th>Type</th><th>Role</th></tr></thead>
        <tbody>
          {cols.map(c => (
            <tr key={c.name}>
              <td>{c.name}</td>
              <td style={{ color: 'var(--text-dim)' }}>{c.dtype}</td>
              <td>
                <select className="model-select" style={{ maxWidth: 140 }} value={roleFor(c.name)}
                  disabled={loading} aria-label={`Role for column ${c.name}`}
                  onChange={e => setEdits(p => ({ ...p, [c.name]: e.target.value }))}>
                  {ROLE_OPTIONS.map(r => <option key={r} value={r}>{r || '—'}</option>)}
                </select>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      <button className="chip" style={{ marginTop: 8 }} disabled={loading || Object.keys(edits).length === 0}
        onClick={() => onApply(edits)}>Apply roles</button>
    </div>
  );
}

export {
  QcCard,
  ReviewCard,
  SkillsPanel,
  NcaSubjectTable,
  DoseSummaryTable,
  DescriptiveStatsTable,
  NcaTable,
  LlmSettings,
  SpaghettiChart,
  NcaLzPlot,
  RolesEditor,
};
