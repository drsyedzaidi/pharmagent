import { useCallback, useRef, useState } from 'react';
import { evidenceRelayoutAll, evidenceRelayoutSource } from './api';
import { bulkSummary, itemFromEntry, outcomeText, rebuildErrorText, type RebuildItem } from './layoutRebuild';
import type { EvidenceSource, RelayoutOutcome } from './types';

type Named = Pick<EvidenceSource, 'id' | 'filename'>;

export interface RelayoutControls {
  /** Per source id: running, done (with the outcome) or failed. Cleared by nothing: the next run replaces it. */
  items: Record<string, RebuildItem>;
  /** One polite sentence about the latest call ('' before the first). */
  summary: string;
  /** A bulk call failed as a whole (a single source's failure is in its item). */
  error: string;
  running: boolean;
  tick: number;
  /** Rebuild one source; resolves to the outcome, or null when the call failed. */
  one: (src: Named) => Promise<RelayoutOutcome | null>;
  /** Rebuild every source without a layout (the server takes at most its cap per call). */
  all: (candidates: readonly Named[]) => Promise<void>;
}

/** Rebuild state for the Library list and the page-image notice. Every finished call asks `onDone` to
 *  re-read the ledger (the sources' layout status) and PharmState (the audit entry). */
export function useRelayout(sessionId: string, onDone: () => void): RelayoutControls {
  const [items, setItems] = useState<Record<string, RebuildItem>>({});
  const [summary, setSummary] = useState('');
  const [error, setError] = useState('');
  const [tick, setTick] = useState(0);
  const [active, setActive] = useState(0);
  const inflight = useRef(new Set<string>());
  const bulkRunning = useRef(false);
  const patch = useCallback((id: string, item: RebuildItem | null) => setItems(prev => {
    const next = { ...prev };
    if (item) next[id] = item; else delete next[id];
    return next;
  }), []);

  const one = useCallback(async (src: Named): Promise<RelayoutOutcome | null> => {
    if (inflight.current.has(src.id)) return null;            // a double click never starts a second parse
    inflight.current.add(src.id);
    setActive(n => n + 1);
    setError('');
    patch(src.id, { phase: 'running', text: 'Rebuilding layout…' });
    setSummary(`Rebuilding layout for ${src.filename}…`);
    try {
      const res = await evidenceRelayoutSource(sessionId, src.id);
      const text = outcomeText(res.outcome);
      patch(src.id, { phase: 'done', outcome: res.outcome, text });
      setSummary(`${src.filename}: ${text}`);
      return res.outcome;
    } catch (e) {
      const text = `Could not rebuild: ${rebuildErrorText(e)}`;
      patch(src.id, { phase: 'error', text });
      setSummary(`${src.filename}: ${text}`);
      return null;
    } finally {
      inflight.current.delete(src.id);
      setActive(n => n - 1);
      onDone();
    }
  }, [sessionId, onDone, patch]);

  const all = useCallback(async (candidates: readonly Named[]) => {
    if (bulkRunning.current || candidates.length === 0) return;
    bulkRunning.current = true;
    setActive(n => n + 1);
    setError('');
    setTick(0);
    for (const c of candidates) patch(c.id, { phase: 'running', text: 'Waiting for a parser slot…' });
    setSummary(`Rebuilding layout for ${candidates.length} source${candidates.length === 1 ? '' : 's'}…`);
    try {
      const { call, bulk } = await evidenceRelayoutAll(sessionId, setTick);
      const picked = new Set(call.source_ids);
      for (const c of candidates) if (!picked.has(c.id)) patch(c.id, null);   // beyond the cap: left for the next call
      for (const entry of bulk.results) patch(entry.source_id, itemFromEntry(entry));
      setSummary(`Rebuild all: ${bulkSummary(bulk.counts, call.remaining)}`);
    } catch (e) {
      for (const c of candidates) patch(c.id, null);
      const text = `Could not rebuild: ${rebuildErrorText(e)}`;
      setError(text);
      setSummary(text);
    } finally {
      bulkRunning.current = false;
      setActive(n => n - 1);
      onDone();
    }
  }, [sessionId, onDone, patch]);

  return { items, summary, error, running: active > 0, tick, one, all };
}
