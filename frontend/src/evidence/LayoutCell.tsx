import { AlertTriangle, CheckCircle, Loader2, OctagonAlert, XCircle } from 'lucide-react';
import { LAYOUT_LABEL, rebuildName, type RebuildItem } from './layoutRebuild';
import type { EvidenceSource, LayoutStatus } from './types';

const PILL: Record<LayoutStatus, string> = { ready: 'approved', missing: 'pending', text_changed: 'rejected', unreadable: 'rejected' };
const ICON = { ready: CheckCircle, missing: AlertTriangle, text_changed: OctagonAlert, unreadable: XCircle };

/** What a finished or failed rebuild says, with an icon and the word for its kind (never colour alone). */
export function RebuildNote({ item }: { item: RebuildItem }) {
  if (item.phase === 'running') {
    return <p className="ev-rebuild-note"><Loader2 size={12} className="spin" aria-hidden="true" /> <span>{item.text}</span></p>;
  }
  const bad = item.phase === 'error' || item.outcome === 'text_changed' || item.outcome === 'unreadable';
  const Icon = item.phase === 'error' ? XCircle : bad ? OctagonAlert : CheckCircle;
  return (
    <p className={`ev-rebuild-note ${bad ? 'bad' : 'ok'}`}>
      <Icon size={12} aria-hidden="true" /> <span>{item.text}</span>
    </p>
  );
}

/** One source's layout status in the Library list: text plus an icon, the reason it is not ready, and a
 *  rebuild button only where a rebuild can help (a missing layout). */
export function LayoutCell({ source, item, disabled, onRebuild }: {
  source: EvidenceSource; item: RebuildItem | undefined; disabled: boolean; onRebuild: (s: EvidenceSource) => void;
}) {
  const status = source.layout_status;
  const Icon = ICON[status];
  const running = item?.phase === 'running';
  return (
    <div className="ev-layout-cell">
      <span className={`pill ${PILL[status]}`}><Icon size={12} aria-hidden="true" /> {LAYOUT_LABEL[status]}</span>
      {status !== 'ready' && source.layout_reason && <p className="ev-prov ev-layout-reason">{source.layout_reason}</p>}
      {status === 'missing' && (
        <button type="button" className="btn btn-ghost ev-rebuild-btn" disabled={disabled || running}
          aria-label={rebuildName(source)} onClick={() => onRebuild(source)}>
          Rebuild layout
        </button>
      )}
      {item && <RebuildNote item={item} />}
    </div>
  );
}
