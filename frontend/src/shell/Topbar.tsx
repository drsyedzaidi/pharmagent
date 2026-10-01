import { useRef, useState } from 'react';
import type { ReactNode } from 'react';
import { FlaskConical, KeyRound } from 'lucide-react';
import { Popover } from './Popover';

/** 52px application header: brand, session breadcrumb, backend/model status,
 *  the model picker (slot filled by App's LlmSettings), the API-token field
 *  and — on narrow screens — the review-drawer toggle. Presentational only:
 *  every handler and the LLM panel come from App. */
export function Topbar({
  sessionId, title, healthy, statusLabel,
  llmOpen, onLlmToggle, onLlmClose, llmPanel,
  token, onTokenChange, review,
}: {
  sessionId: string | null;
  title: string;
  healthy: boolean | null;
  statusLabel: string;
  llmOpen: boolean;
  onLlmToggle: () => void;
  onLlmClose: () => void;
  llmPanel: ReactNode;
  token: string;
  onTokenChange: (v: string) => void;
  review?: { open: boolean; pending: boolean; onToggle: () => void };
}) {
  const modelRef = useRef<HTMLButtonElement>(null);
  const tokenRef = useRef<HTMLButtonElement>(null);
  const [tokenOpen, setTokenOpen] = useState(false);

  return (
    <header className="topbar">
      <div className="topbar-logo">
        <FlaskConical size={18} aria-hidden="true" />
        <span>PharmAgent</span>
      </div>
      <span className="topbar-divider" aria-hidden="true" />
      <nav aria-label="Breadcrumb" className="breadcrumb">
        <span>Sessions</span>
        <span aria-hidden="true">/</span>
        <span className="breadcrumb-title">{title}</span>
        {sessionId && (
          <span className="id-chip" title={sessionId}>{sessionId.slice(0, 8)}</span>
        )}
      </nav>

      <div className="topbar-right">
        <div className="status-pill">
          <span
            className="status-dot"
            style={{ background: healthy === false ? 'var(--red)' : 'var(--green)' }}
            aria-hidden="true"
          />
          <span>{statusLabel}</span>
        </div>

        <div style={{ position: 'relative' }}>
          <button
            ref={modelRef}
            type="button"
            className="btn"
            aria-haspopup="dialog"
            aria-expanded={llmOpen}
            title="Change the language model (local / ChatGPT / Claude)"
            onClick={onLlmToggle}
          >
            Model ▾
          </button>
          <Popover open={llmOpen} onClose={onLlmClose} anchorRef={modelRef} align="right" ariaLabel="Language model">
            {llmPanel}
          </Popover>
        </div>

        <div style={{ position: 'relative' }}>
          <button
            ref={tokenRef}
            type="button"
            className="btn btn-icon topbar-icon-btn"
            aria-label="API token"
            aria-haspopup="dialog"
            aria-expanded={tokenOpen}
            title="API token"
            onClick={() => setTokenOpen(o => !o)}
          >
            <KeyRound size={14} aria-hidden="true" />
          </button>
          <Popover open={tokenOpen} onClose={() => setTokenOpen(false)} anchorRef={tokenRef} align="right" ariaLabel="API token">
            <div style={{ display: 'flex', flexDirection: 'column', gap: 6, width: 280 }}>
              <label htmlFor="api-token" style={{ fontSize: 12, fontWeight: 600, color: 'var(--text-h)' }}>API token</label>
              <input
                id="api-token"
                className="token-input"
                type="password"
                placeholder="API token (optional)"
                value={token}
                onChange={e => onTokenChange(e.target.value)}
                style={{ width: '100%' }}
                autoComplete="off"
              />
              <span style={{ fontSize: 12, color: 'var(--text-dim)' }}>
                Bearer token — only needed when the backend sets an API token
              </span>
            </div>
          </Popover>
        </div>

        {review && (
          <button
            id="review-toggle"
            type="button"
            className="btn review-toggle"
            aria-expanded={review.open}
            aria-controls="review-panel"
            onClick={review.onToggle}
          >
            {review.pending && <span className="review-toggle-dot" aria-hidden="true" />}
            {review.pending ? 'Review: decision needed' : 'Review'}
          </button>
        )}
      </div>
    </header>
  );
}
