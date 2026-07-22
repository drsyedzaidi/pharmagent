import type {
  Session, ChatResponse, WorkflowResponse, AuditEntry, PharmState, PkModelDef, JobResult,
  ReviewLoopResult, SkillDef, VariablesResponse, FlexplotSpec, FlexplotData,
} from './types';

const BASE = '/api';

const TOKEN_KEY = 'pharmagent_token';
let _token = (typeof localStorage !== 'undefined' && localStorage.getItem(TOKEN_KEY)) || '';

export function setToken(t: string): void {
  _token = t.trim();
  if (typeof localStorage !== 'undefined') {
    if (_token) localStorage.setItem(TOKEN_KEY, _token);
    else localStorage.removeItem(TOKEN_KEY);
  }
}
export function getToken(): string {
  return _token;
}

function authHeaders(extra?: Record<string, string>): Record<string, string> {
  const h: Record<string, string> = { ...extra };
  if (_token) h.Authorization = `Bearer ${_token}`;
  return h;
}

async function req<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(BASE + path, { ...init, headers: authHeaders(init?.headers as Record<string, string>) });
  if (!res.ok) {
    const text = await res.text().catch(() => res.statusText);
    throw new Error(text || `HTTP ${res.status}`);
  }
  return res.json() as Promise<T>;
}

async function download(path: string, filename: string): Promise<void> {
  const res = await fetch(BASE + path, { headers: authHeaders() });
  if (!res.ok) throw new Error((await res.text().catch(() => res.statusText)) || `HTTP ${res.status}`);
  const blob = await res.blob();
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url; a.download = filename;
  document.body.appendChild(a); a.click(); a.remove();
  URL.revokeObjectURL(url);
}

export const api = {
  health: () => req<{ status: string; llm: string }>('/health'),

  createSession: () => req<Session>('/sessions', { method: 'POST' }),

  uploadDataset: (sid: string, file: File): Promise<{ dataset_id: string; metadata: Record<string, unknown> }> => {
    const fd = new FormData();
    fd.append('file', file);
    return req(`/sessions/${sid}/upload`, { method: 'POST', body: fd });
  },

  chat: (sid: string, message: string): Promise<ChatResponse> =>
    req(`/sessions/${sid}/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message }),
    }),

  startWorkflow: (sid: string, path: string, workflow = 'nca_full'): Promise<WorkflowResponse> =>
    req(`/sessions/${sid}/workflow/start`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ workflow, params: { path } }),
    }),

  resumeWorkflow: (sid: string, approve: boolean): Promise<WorkflowResponse> =>
    req(`/sessions/${sid}/workflow/resume`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ approve }),
    }),

  // Approving a gate runs every remaining step in one call. When that remainder
  // is a population fit (poppk_full), ask for a job id and poll it rather than
  // holding the request open for minutes.
  resumeWorkflowAsync: (sid: string, reason = ''):
    Promise<{ job_id: string; status: string; kind: string }> =>
    req(`/sessions/${sid}/workflow/resume`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ approve: true, reason, background: true }),
    }),

  getState: (sid: string): Promise<PharmState> =>
    req(`/sessions/${sid}/state`),

  variables: (sid: string): Promise<VariablesResponse> =>
    req(`/sessions/${sid}/variables`),

  flexplot: (sid: string, body: FlexplotSpec): Promise<{ state: { flexplot_data: FlexplotData | null } }> =>
    req(`/sessions/${sid}/flexplot`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    }),

  setRoles: (sid: string, overrides: Record<string, string>):
    Promise<{ detected_roles: Record<string, string>; state: PharmState }> =>
    req(`/sessions/${sid}/roles`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ overrides }),
    }),

  listPkModels: (): Promise<{ models: PkModelDef[] }> =>
    req('/pk_models'),

  runPkModel: (sid: string, body: { model_key?: string; compare?: boolean; models?: string[] }):
    Promise<{ agent: string; summary: string; state: PharmState; audit_ok: boolean }> =>
    req(`/sessions/${sid}/pkmodel`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    }),

  simulate: (sid: string, body: {
    model_key?: string; dose: number; tau: number; n_doses: number; tmax?: number;
  }): Promise<{ agent: string; summary: string; state: PharmState; audit_ok: boolean }> =>
    req(`/sessions/${sid}/simulate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    }),

  vpc: (sid: string): Promise<{ agent: string; summary: string; state: PharmState; audit_ok: boolean }> =>
    req(`/sessions/${sid}/vpc`, { method: 'POST' }),

  diagnostics: (sid: string): Promise<{ agent: string; summary: string; state: PharmState; audit_ok: boolean }> =>
    req(`/sessions/${sid}/diagnostics`, { method: 'POST' }),

  forest: (sid: string): Promise<{ agent: string; summary: string; state: PharmState; audit_ok: boolean }> =>
    req(`/sessions/${sid}/forest`, { method: 'POST' }),

  nlme: (sid: string, body: { method: string; model_key?: string; error_model?: string }):
    Promise<{ job_id: string; status: string; kind: string }> =>
    req(`/sessions/${sid}/nlme`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    }),

  scm: (sid: string, body: { model_key?: string; error_model?: string; iiv_params?: string[] }):
    Promise<{ job_id: string; status: string; kind: string }> =>
    req(`/sessions/${sid}/scm`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    }),

  simest: (sid: string, body: {
    confirm: boolean;
    design: {
      n_subjects: number; obs_t: number[]; dose?: number; dose_per_kg?: number;
      n_doses?: number; tau?: number; wt_mean?: number; wt_cv_pct?: number; lloq?: number;
    };
    n_rep?: number; params?: string[]; ci_target_pct?: number; method?: string;
  }): Promise<{ job_id: string; status: string; kind: string }> =>
    req(`/sessions/${sid}/simest`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    }),

  engineComparison: (sid: string, body: {
    candidates?: { model_key: string; iiv_params?: string[]; error_model?: string; method?: string }[];
    engines?: string[];
    model_key?: string;
  }): Promise<{ job_id: string; status: string; kind: string }> =>
    req(`/sessions/${sid}/engine_comparison`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    }),

  forecast: (sid: string, body: {
    dose: number; tau: number; measured: { time: number; conc: number }[];
    wt?: number; target?: number; target_metric?: string;
  }): Promise<{ agent: string; summary: string; state: PharmState; audit_ok: boolean }> =>
    req(`/sessions/${sid}/forecast`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    }),

  refitLz: (sid: string, body: {
    subject: string;
    selected_times: number[];
    selected_concs: number[];
  }): Promise<{
    lambda_z: number; lambda_z_intercept: number;
    t_half: number; r2_adj: number; n_pts: number;
    lz_x: number[]; lz_y: number[];
    fit_x: number[]; fit_y: number[];
  }> =>
    req(`/sessions/${sid}/nca/refit_lz`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    }),

  getJob: (sid: string, jobId: string):
    Promise<{ status: string; kind: string; result: JobResult | null; error: string | null }> =>
    req(`/sessions/${sid}/jobs/${jobId}`),

  // Poll a background job to completion. onTick fires each poll with elapsed seconds.
  pollJob: async <T = JobResult>(sid: string, jobId: string,
                  onTick?: (elapsedSec: number) => void): Promise<T> => {
    const start = Date.now();
    for (;;) {
      const j = await api.getJob(sid, jobId);
      if (j.status === 'done' && j.result) return j.result as unknown as T;
      if (j.status === 'error') throw new Error(j.error || 'job failed');
      onTick?.(Math.round((Date.now() - start) / 1000));
      await new Promise(r => setTimeout(r, 1500));
    }
  },

  cdiscUrl: (sid: string): string => `${BASE}/sessions/${sid}/cdisc`,
  downloadCdisc: (sid: string): Promise<void> =>
    download(`/sessions/${sid}/cdisc`, `cdisc_adam_${sid}.zip`),

  generateReport: (sid: string): Promise<{ state: PharmState; result: { report_path: string } }> =>
    req(`/sessions/${sid}/report`, { method: 'POST' }),

  exportCsv: (sid: string, kind: string): Promise<void> =>
    download(`/sessions/${sid}/export/${kind}`, `${kind}_${sid}.csv`),

  exportControl: (sid: string, kind: 'nonmem' | 'mrgsolve'): Promise<void> =>
    download(`/sessions/${sid}/export/control/${kind}`,
             `${kind}_${sid}.${kind === 'nonmem' ? 'ctl' : 'cpp'}`),

  downloadReportFile: (sid: string, reportPath: string): Promise<void> => {
    const filename = reportPath.split('/').pop() ?? 'report.docx';
    return download(`/sessions/${sid}/report/${filename}`, filename);
  },

  doseSweep: (sid: string, body: {
    doses?: number[]; tau: number; n_doses: number; tmax?: number;
  }): Promise<{ agent: string; summary: string; state: PharmState; audit_ok: boolean }> =>
    req(`/sessions/${sid}/dosesweep`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    }),

  review: (sid: string, body?: { goal?: string; max_iter?: number }):
    Promise<ReviewLoopResult> =>
    req(`/sessions/${sid}/review`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body ?? {}),
    }),

  listSkills: (): Promise<{ skills: SkillDef[] }> => req('/skills'),

  captureSkill: (sid: string, body: { name: string; description?: string; goal?: string }):
    Promise<{ skill: SkillDef }> =>
    req(`/sessions/${sid}/capture-skill`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    }),

  runSkill: (name: string, datasetPath: string):
    Promise<{ skill: string; session_id: string; executed: { tool: string; status: string }[]; state: PharmState }> =>
    req(`/skills/${encodeURIComponent(name)}/run`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ dataset_path: datasetPath }),
    }),

  deleteSkill: (name: string): Promise<{ deleted: string }> =>
    req(`/skills/${encodeURIComponent(name)}`, { method: 'DELETE' }),

  skillMarkdown: (name: string): Promise<void> =>
    download(`/skills/${encodeURIComponent(name)}/markdown`, `${name}.SKILL.md`),

  getAudit: (sid: string): Promise<{ entries: AuditEntry[]; verified: boolean; count: number }> =>
    req(`/sessions/${sid}/audit`),

  downloadReport: (sid: string, reportPath: string): string => {
    const filename = reportPath.split('/').pop() ?? 'report.docx';
    return `${BASE}/sessions/${sid}/report/${filename}`;
  },
};
