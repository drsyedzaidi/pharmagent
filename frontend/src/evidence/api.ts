/** Evidence endpoint helpers (/api/sessions/{sid}/evidence/...). They reuse api.ts's
 *  authenticated `req` / `download` and `api.pollJob`; kept here so api.ts stays small.
 *  POST /screen and /extract answer with a job handle when a real LLM is configured,
 *  so those two helpers poll it and always resolve to the tool envelope. */
import { api, download, req } from '../api';
import type {
  ConfigResponse, CriteriaConfig, Criterion, EvidenceSnapshot, ExportResult, ExtractResult, PageText,
  ProposeResponse, RecordDecideResponse, RecordDecision, RecordFilter, ScreenDecideResponse, ScreenDecision,
  ScreenResult, TemplateBody, TemplateConfig, ToolEnvelope, UploadResponse, VerifyRecordsResult,
} from './types';

const ev = (sid: string) => `/sessions/${encodeURIComponent(sid)}/evidence`;
const json = (body: unknown): RequestInit => ({
  method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body),
});

async function propose<R>(sid: string, path: string, sourceIds?: string[],
                          onTick?: (elapsedSec: number) => void): Promise<ToolEnvelope<R>> {
  const res = await req<ProposeResponse<R>>(ev(sid) + path, json(sourceIds ? { source_ids: sourceIds } : {}));
  if ('job_id' in res) return api.pollJob<ToolEnvelope<R>>(sid, res.job_id, onTick);
  return res;
}

export function evidenceUpload(sid: string, file: File, licenceNote = ''): Promise<UploadResponse> {
  const fd = new FormData();
  fd.append('file', file);
  if (licenceNote.trim()) fd.append('licence_note', licenceNote.trim());
  return req(ev(sid) + '/upload', { method: 'POST', body: fd });
}

export function evidenceSnapshot(sid: string, filter: RecordFilter = {}): Promise<EvidenceSnapshot> {
  const q = new URLSearchParams();
  if (filter.status) q.set('status', filter.status);
  if (filter.verify) q.set('verify', filter.verify);
  if (filter.flagged !== undefined) q.set('flagged', String(filter.flagged));
  const qs = q.toString();
  return req(ev(sid) + (qs ? `?${qs}` : ''));
}

export const evidenceCriteria = (sid: string, criteria: Criterion[]): Promise<ConfigResponse<CriteriaConfig>> =>
  req(ev(sid) + '/criteria', json({ criteria }));

export const evidenceTemplate = (sid: string, body: TemplateBody): Promise<ConfigResponse<TemplateConfig>> =>
  req(ev(sid) + '/template', json(body));

export const evidenceScreen = (sid: string, sourceIds?: string[], onTick?: (s: number) => void) =>
  propose<ScreenResult>(sid, '/screen', sourceIds, onTick);

export const evidenceDecideScreen = (sid: string, decisions: ScreenDecision[]): Promise<ScreenDecideResponse> =>
  req(ev(sid) + '/screen/decide', json({ decisions }));

export const evidenceExtract = (sid: string, sourceIds?: string[], onTick?: (s: number) => void) =>
  propose<ExtractResult>(sid, '/extract', sourceIds, onTick);

export const evidenceVerify = (sid: string, recordIds?: string[]): Promise<ToolEnvelope<VerifyRecordsResult>> =>
  req(ev(sid) + '/records/verify', json(recordIds ? { record_ids: recordIds } : {}));

export const evidenceDecideRecords = (sid: string, decisions: RecordDecision[]): Promise<RecordDecideResponse> =>
  req(ev(sid) + '/records/decide', json({ decisions }));

export const evidenceExport = (sid: string): Promise<ToolEnvelope<ExportResult>> =>
  req(ev(sid) + '/export', json({}));

/** Download an exported CSV by the basename of its server path (the route serves only this session's exports). */
export function evidenceDownload(sid: string, path: string): Promise<void> {
  const name = path.split(/[\\/]/).pop() ?? '';
  return download(`${ev(sid)}/dataset/${encodeURIComponent(name)}`, name);
}

export const evidencePage = (sid: string, sourceId: string, page: number): Promise<PageText> =>
  req(`${ev(sid)}/sources/${encodeURIComponent(sourceId)}/pages/${page}`);
