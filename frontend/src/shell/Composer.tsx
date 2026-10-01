import type { KeyboardEvent, ReactNode, RefObject } from 'react';
import { Loader2, Plus, Send } from 'lucide-react';

/** The chat composer under the document: background-job note, the
 *  Actions button (anchor for the ActionsMenu popover passed as children),
 *  the ask textarea and the Send button. All state and handlers stay in App;
 *  this only lays them out. */
export function Composer({
  value, onChange, onKeyDown, onSend, textDisabled, sendDisabled, loading, jobNote,
  actionsOpen, onToggleActions, actionsRef, children,
}: {
  value: string;
  onChange: (v: string) => void;
  onKeyDown: (e: KeyboardEvent<HTMLTextAreaElement>) => void;
  onSend: () => void;
  textDisabled: boolean;
  sendDisabled: boolean;
  loading: boolean;
  jobNote: string;
  actionsOpen: boolean;
  onToggleActions: () => void;
  actionsRef: RefObject<HTMLButtonElement | null>;
  children: ReactNode;
}) {
  return (
    <div className="composer">
      {jobNote && (
        <div className="job-note" role="status">
          <span className="job-spinner" /> {jobNote}
        </div>
      )}
      <div className="composer-row">
        <button
          ref={actionsRef}
          type="button"
          className="btn actions-btn"
          aria-haspopup="dialog"
          aria-expanded={actionsOpen}
          aria-controls="actions-pop"
          onClick={onToggleActions}
        >
          <Plus size={14} aria-hidden="true" /> Actions
        </button>
        {/* popover sits right after its anchor in DOM (Tab order); absolute, so layout is unchanged */}
        {children}
        <label className="sr-only" htmlFor="ask">Ask the agents</label>
        <textarea
          id="ask"
          rows={1}
          placeholder="Ask the agents, e.g. refit λz for a subject with its last four points"
          value={value}
          onChange={e => onChange(e.target.value)}
          onKeyDown={onKeyDown}
          disabled={textDisabled}
        />
        <button
          type="button"
          className="btn btn-primary btn-icon"
          aria-label="Send"
          title="Send"
          disabled={sendDisabled}
          onClick={onSend}
        >
          {loading
            ? <Loader2 size={14} className="spin" aria-hidden="true" />
            : <Send size={14} aria-hidden="true" />}
        </button>
      </div>
      <p className="composer-hint">Agents propose; you approve. Expensive runs start only after you confirm them.</p>
    </div>
  );
}
