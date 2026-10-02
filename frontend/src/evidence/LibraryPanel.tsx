import { useRef, useState } from 'react';
import type { ChangeEvent, DragEvent } from 'react';
import { AlertTriangle, CheckCircle, FileText, Loader2, Upload, XCircle } from 'lucide-react';
import { hhmm } from '../shell/format';
import { evidenceUpload } from './api';
import { errorMessage, fmtBytes, MAX_PDF_BYTES, PARSE_LABEL, shaPrefix } from './format';
import type { EvidenceSource, ParseStatus, UploadResponse } from './types';

type UploadState = 'queued' | 'uploading' | 'done' | 'duplicate' | 'error';
interface UploadItem { key: string; name: string; size: number; state: UploadState; note: string }

const STATE_WORD: Record<UploadState, string> = {
  queued: 'queued', uploading: 'uploading', done: 'added', duplicate: 'already here', error: 'rejected',
};

function ParsePill({ status }: { status: ParseStatus }) {
  const cls = status === 'parsed' ? 'approved' : status === 'no_text' ? 'pending' : 'rejected';
  const Icon = status === 'parsed' ? CheckCircle : status === 'no_text' ? AlertTriangle : XCircle;
  return (
    <span className={`pill ${cls}`}>
      <Icon size={12} aria-hidden="true" /> {PARSE_LABEL[status]}
    </span>
  );
}

function UploadIcon({ state }: { state: UploadState }) {
  if (state === 'uploading') return <Loader2 size={13} className="spin" aria-hidden="true" />;
  if (state === 'done') return <CheckCircle size={13} className="ev-ok" aria-hidden="true" />;
  if (state === 'duplicate') return <CheckCircle size={13} className="ev-dim" aria-hidden="true" />;
  if (state === 'error') return <XCircle size={13} className="ev-bad" aria-hidden="true" />;
  return <FileText size={13} className="ev-dim" aria-hidden="true" />;
}

/** Library tab: PDF drop zone (sequential uploads with per-file status) and the
 *  session's source list. Titles and filenames come from the document and are
 *  rendered as text only. */
export function LibraryPanel({ sessionId, sources, disabled, onUploaded }: {
  sessionId: string;
  sources: EvidenceSource[];
  disabled: boolean;
  /** Called once per batch after the last upload settles (refresh snapshot + state). */
  onUploaded: (last: UploadResponse | null) => void;
}) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [drag, setDrag] = useState(false);
  const [licence, setLicence] = useState('');
  const [items, setItems] = useState<UploadItem[]>([]);
  const [running, setRunning] = useState(false);

  const patch = (key: string, p: Partial<UploadItem>) =>
    setItems(prev => prev.map(it => (it.key === key ? { ...it, ...p } : it)));

  async function uploadAll(files: File[]) {
    if (files.length === 0 || running) return;
    const batch = files.map((f, i) => ({
      file: f,
      item: { key: `${Date.now()}-${i}-${f.name}`, name: f.name, size: f.size, state: 'queued' as UploadState, note: '' },
    }));
    setItems(batch.map(b => b.item));
    setRunning(true);
    let last: UploadResponse | null = null;
    for (const { file, item } of batch) {
      if (!file.name.toLowerCase().endsWith('.pdf')) { patch(item.key, { state: 'error', note: 'PDF only' }); continue; }
      if (file.size > MAX_PDF_BYTES) { patch(item.key, { state: 'error', note: 'exceeds 25 MB limit' }); continue; }
      patch(item.key, { state: 'uploading' });
      try {
        const res = await evidenceUpload(sessionId, file, licence);
        last = res;
        patch(item.key, res.duplicate
          ? { state: 'duplicate', note: `already in this session as ${res.source.id}` }
          : { state: 'done', note: `${res.source.pages} pages · ${PARSE_LABEL[res.source.parse_status]}` });
      } catch (e) {
        patch(item.key, { state: 'error', note: errorMessage(e) });
      }
    }
    setRunning(false);
    onUploaded(last);
  }

  const onPick = (e: ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(e.target.files ?? []);
    e.target.value = '';
    void uploadAll(files);
  };
  const onDrop = (e: DragEvent) => {
    e.preventDefault();
    setDrag(false);
    if (!disabled) void uploadAll(Array.from(e.dataTransfer.files));
  };
  const blocked = disabled || running;

  return (
    <div className="ev-panel">
      <input ref={inputRef} type="file" accept=".pdf,application/pdf" multiple hidden onChange={onPick} />
      <div className="ev-upload-row">
        <button type="button" className={`upload-zone ev-drop ${drag ? 'drag' : ''}`} disabled={blocked}
          aria-describedby="ev-drop-hint"
          onDragOver={e => { e.preventDefault(); if (!blocked) setDrag(true); }}
          onDragLeave={() => setDrag(false)} onDrop={onDrop}
          onClick={() => inputRef.current?.click()}>
          <Upload size={20} className="ev-dim" aria-hidden="true" />
          <span className="upload-label"><strong>Add publications</strong> — click or drop PDFs</span>
          <span id="ev-drop-hint" className="ev-hint">PDF only · up to 25 MB and 300 pages each · password-protected files are refused</span>
        </button>
        <label className="ev-field ev-licence">
          <span className="ev-label">Licence note (optional, applies to this batch)</span>
          <input className="ev-input" type="text" value={licence} maxLength={500} disabled={blocked}
            placeholder="e.g. CC BY 4.0, PMC open access" onChange={e => setLicence(e.target.value)} />
        </label>
      </div>

      {items.length > 0 && (
        <ul className="ev-uploads" aria-label="Upload status" aria-live="polite">
          {items.map(it => (
            <li key={it.key} className={`ev-upload ${it.state}`}>
              <UploadIcon state={it.state} />
              <span className="ev-upload-name">{it.name}</span>
              <span className="ev-dim">{fmtBytes(it.size)}</span>
              <span className="ev-upload-state">{STATE_WORD[it.state]}</span>
              {it.note && <span className="ev-upload-note">{it.note}</span>}
            </li>
          ))}
        </ul>
      )}

      {sources.length === 0 ? (
        <p className="ev-empty">No publications yet. Uploaded PDFs are parsed on the server; nothing is read by a model until you run screening.</p>
      ) : (
        <table className="ev-table">
          <caption className="sr-only">Publications in this session</caption>
          <thead>
            <tr><th scope="col">Publication</th><th scope="col" className="num">Pages</th><th scope="col">SHA-256</th><th scope="col">Parse</th></tr>
          </thead>
          <tbody>
            {sources.map(s => (
              <tr key={s.id}>
                <td>
                  <div className="ev-src-name">{s.filename}</div>
                  {s.title && <div className="ev-src-title">{s.title}</div>}
                  <div className="ev-prov">
                    <code>{s.id}</code> · uploaded by {s.uploaded_by} · {hhmm(s.uploaded_at)}
                    {s.decimal_style === ',' && ' · decimal comma'}
                    {s.licence_note && <> · licence: {s.licence_note}</>}
                  </div>
                </td>
                <td className="num">{s.pages}</td>
                <td><span className="id-chip" title={s.sha256}>{shaPrefix(s.sha256)}</span></td>
                <td><ParsePill status={s.parse_status} /></td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
}
