/** Which records bulk approval may offer. Pure, no React.
 *
 *  Bulk approval covers only verified, unflagged proposals whose PDF has a usable page layout. A source whose
 *  layout is missing, changed or unreadable has no outlined image for its values (the server marks them
 *  compare_urgent), so each of its records is opened and compared one by one, never swept up in a batch. An
 *  unknown source counts as unusable: nothing passes silently. */
import type { EvidenceSource, RecordRow } from './types';

type Layout = Pick<EvidenceSource, 'id' | 'layout_status'>;

export const layoutUsable = (source: Pick<EvidenceSource, 'layout_status'> | null | undefined): boolean =>
  source?.layout_status === 'ready';

/** Verified, unflagged, still proposed, and cited from a source with a ready layout. */
export const bulkEligible = (r: RecordRow, sources: ReadonlyMap<string, Layout>): boolean =>
  r.status === 'proposed' && r.verify_status === 'pass' && r.flags.length === 0 && layoutUsable(sources.get(r.source_id));

/** Records that would be bulk-eligible but for their source's layout: they are left out, and say why. */
export const layoutLeftOut = (rows: readonly RecordRow[], sources: ReadonlyMap<string, Layout>): number =>
  rows.filter(r => r.status === 'proposed' && r.verify_status === 'pass' && r.flags.length === 0
    && !layoutUsable(sources.get(r.source_id))).length;

export const leftOutText = (n: number): string => n === 0 ? ''
  : `${n} verified record${n === 1 ? ' is' : 's are'} left out because the page layout of ${n === 1 ? 'its' : 'their'} PDF `
    + 'is missing, changed or unreadable: open each one and compare it with the PDF.';
