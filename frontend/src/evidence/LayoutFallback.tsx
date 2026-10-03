import { AlertTriangle, Loader2, OctagonAlert } from 'lucide-react';
import { rebuildName } from './layoutRebuild';
import { RebuildNote } from './LayoutCell';
import type { EvidenceSource } from './types';
import type { RelayoutControls } from './useRelayout';

const EXPLAIN = {
  missing: 'Parsed before page positions were stored: the whole page is shown and the cited value is not outlined. '
    + 'Rebuild the layout to outline it.',
  text_changed: 'The page text of this PDF now parses differently, so stored page positions cannot be tied to the text the records cite. '
    + 'No rebuild is offered: the whole page stays. Upload the PDF to a new session, extract again there, and compare each value with the PDF itself.',
  unreadable: 'This PDF cannot be parsed again, so no page positions can be stored. The whole page stays: compare each value with the PDF itself.',
} as const;

/** The notice that goes with a whole-page image whose source has no usable layout: what happened, what the
 *  reviewer can do (a rebuild button only for a missing layout), and how the rebuild is going. A polite live
 *  region (kept out of the layout) is always mounted, so a result that arrives as the notice goes away is still
 *  announced; the visible result stays until the next rebuild or the record closes. */
export function LayoutFallback({ source, rebuild, onRebuilt }: {
  source: EvidenceSource; rebuild: RelayoutControls; onRebuilt: () => void;
}) {
  const status = source.layout_status;
  const item = rebuild.items[source.id];
  const running = item?.phase === 'running';
  const Icon = status === 'missing' ? AlertTriangle : OctagonAlert;
  const go = async () => {
    const outcome = await rebuild.one(source);
    if (outcome !== null) onRebuilt();                      // fetch the meta and the image again
  };
  return (
    <>
      <p className="sr-only" role="status">{item ? item.text : ''}</p>
      {status !== 'ready' ? (
        <div className={`ev-notice ev-fallback ${status === 'missing' ? 'warn' : 'error'}`}>
          <Icon size={14} aria-hidden="true" />
          <div className="ev-fallback-body">
            <p><strong>{status === 'missing' ? 'Note:' : 'Check:'}</strong> {EXPLAIN[status]}</p>
            {status === 'missing' && (
              <button type="button" className="btn btn-ghost ev-rebuild-btn" disabled={running || rebuild.running}
                aria-label={`Rebuild layout for this source, ${rebuildName(source).replace('Rebuild layout for ', '')}`}
                onClick={() => void go()}>
                {running ? <><Loader2 size={12} className="spin" aria-hidden="true" /> Rebuilding…</> : 'Rebuild layout for this source'}
              </button>
            )}
            {item && <RebuildNote item={item} />}
          </div>
        </div>
      ) : item && <div className="ev-fallback-done"><RebuildNote item={item} /></div>}
    </>
  );
}
