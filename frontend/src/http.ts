/** Authenticated fetch plumbing shared by api.ts and the evidence helpers: bearer token, JSON requests,
 *  binary GETs and downloads. Re-exported from api.ts. */
export const BASE = '/api';

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

/** A failed request: `message` is the response text (the JSON error envelope), `status` and the
 *  Retry-After seconds (429/503) let callers say how long to wait. */
export class HttpError extends Error {
  status: number;
  retryAfter: number | null;
  constructor(message: string, status: number, retryAfter: number | null) {
    super(message);
    this.status = status;
    this.retryAfter = retryAfter;
  }
}

async function checked(res: Response): Promise<Response> {
  if (res.ok) return res;
  const text = await res.text().catch(() => res.statusText);
  const wait = Number(res.headers.get('Retry-After'));
  throw new HttpError(text || `HTTP ${res.status}`, res.status, Number.isFinite(wait) && wait > 0 ? wait : null);
}

export async function req<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await checked(await fetch(BASE + path, { ...init, headers: authHeaders(init?.headers as Record<string, string>) }));
  return res.json() as Promise<T>;
}

/** GET a binary body with the bearer header (an <img src> cannot send one). */
export async function fetchBlob(path: string, signal?: AbortSignal): Promise<Blob> {
  return (await checked(await fetch(BASE + path, { headers: authHeaders(), signal }))).blob();
}

export async function download(path: string, filename: string): Promise<void> {
  const url = URL.createObjectURL(await fetchBlob(path));
  const a = document.createElement('a');
  a.href = url; a.download = filename;
  document.body.appendChild(a); a.click(); a.remove();
  URL.revokeObjectURL(url);
}
