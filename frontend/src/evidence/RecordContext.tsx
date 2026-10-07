import { OctagonAlert } from 'lucide-react';
import { useEffect, useRef, useState } from 'react';
import type { ReactNode } from 'react';
import { evidencePage } from './api';
import { errorMessage } from './format';
import { contextText, highlight, markLiteral, markScrollTop } from './highlight';
import { compareLabel, imageAnnouncement, imageUnavailableText, urgentFlagCodes } from './imageGeometry';
import { PageImage } from './PageImage';
import type { EvidenceSource, PageText, ParsedTable, RecordRow, Span, TableCell } from './types';
import type { PageImageView } from './usePageImage';
import type { RelayoutControls } from './useRelayout';

/** The marked snippet, with the verified literal in bold at the verifier's token (`start`: where `hit` begins in the
 *  text the token indexes; 0 for a table cell), never at an earlier occurrence of the same digits. */
function Hit({ hit, literal, token, start }: { hit: string; literal: string | null; token?: Span | null; start: number }) {
  const parts = markLiteral(hit, literal, token, start);
  if (!parts) return <>{hit}</>;
  return <>{parts[0]}<strong>{parts[1]}</strong>{parts[2]}</>;
}

/** The parsed table a table locator cites, with the cited cell marked. */
function CellGrid({ table, cell, literal, token }: { table: ParsedTable; cell: TableCell; literal: string | null; token?: Span | null }) {
  return (
    <div className="ev-grid-wrap">
      <table className="ev-grid">
        <caption>{table.caption ?? `Table ${table.index + 1}`} · {table.strategy === 'layout' ? 'unruled (layout) table' : 'ruled table'}</caption>
        <tbody>
          {table.rows.map((row, r) => (
            <tr key={r}>
              {row.map((c, k) => {
                const hit = r === cell.row && k === cell.col;
                const Cell = r === 0 ? 'th' : 'td';
                return (
                  <Cell key={k} scope={r === 0 ? 'col' : undefined} className={hit ? 'ev-cell-hit' : undefined}
                    aria-current={hit ? 'true' : undefined}>
                    {hit ? <mark><Hit hit={c} literal={literal} token={token} start={0} /></mark> : c}
                  </Cell>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

/** Context of one record: the cited page text in a monospace block with the snippet marked
 *  at the verifier's raw span, and for table locators the parsed grid with the cell marked
 *  (a table record whose row is not in the page text is marked in the grid only, never at a
 *  guessed place in the text). Page text is untrusted document content and is rendered as
 *  text nodes only. Opened by the reviewer, it takes focus on its heading; Close or Escape
 *  hands focus back (onClose). */
export function RecordContext({ sessionId, record, filename, view, source, rebuild, decide, autoFocus = false, onClose }: {
  sessionId: string;
  record: RecordRow;
  filename: string;
  /** The record's page image, owned by the records panel (the compare gate reads it too). */
  view: PageImageView;
  source: EvidenceSource | null;
  /** The workspace's rebuild controls (shared with the Library list). */
  rebuild: RelayoutControls;
  /** The record's decision controls, drawn right under the page image so the picture, the compare tick and
   *  Approve are on screen together (the snippet and page text below them are for reference). */
  decide?: ReactNode;
  autoFocus?: boolean;
  onClose?: () => void;
}) {
  const loc = record.locator;
  const [page, setPage] = useState<PageText | null>(null);
  const [error, setError] = useState('');
  const boxRef = useRef<HTMLPreElement>(null);
  const markRef = useRef<HTMLElement>(null);
  const headRef = useRef<HTMLHeadingElement>(null);
  // Urgency is known from the record's own flags, so it survives an image that never loads.
  const urgent = urgentFlagCodes(record.flags).length > 0;
  const decided = record.status !== 'proposed';

  useEffect(() => {
    let alive = true;
    const load = async () => {
      try {
        const p = await evidencePage(sessionId, record.source_id, loc.page);
        if (alive) { setPage(p); setError(''); }
      } catch (e) {
        if (alive) { setPage(null); setError(errorMessage(e)); }
      }
    };
    void load();
    return () => { alive = false; };
  }, [sessionId, record.source_id, loc.page]);

  useEffect(() => { if (autoFocus) headRef.current?.focus(); }, [autoFocus, record.id]);

  const current = page && page.source_id === record.source_id && page.page === loc.page ? page : null;
  const ctx = current ? contextText(current, loc) : null;
  const cellOnly = loc.kind === 'table' && loc.span === null;
  const hl = ctx && !cellOnly ? highlight(ctx.text, loc.span, loc.snippet) : null;
  const table = current && loc.kind === 'table' && loc.cell
    ? current.tables.find(t => t.index === loc.cell?.table) ?? null : null;

  // Scroll the mark into the upper third of the text box only (never the page; no animation).
  useEffect(() => {
    const box = boxRef.current, mark = markRef.current;
    if (!box || !mark) return;
    box.scrollTop = markScrollTop(box.scrollTop, box.getBoundingClientRect().top, mark.getBoundingClientRect().top,
      box.clientHeight);
  }, [hl?.before, record.id]);

  const parts = [
    `page ${loc.page}${current ? ` of ${current.pages}` : ''}`,
    loc.kind,
    loc.literal !== null ? `literal ${loc.literal}` : 'no literal',
    `decimal '${loc.decimal_style}'`,
  ];
  if (ctx?.column !== null && ctx?.column !== undefined) parts.push(`column ${ctx.column + 1}`);
  if (loc.cell) parts.push(`cell r${loc.cell.row} c${loc.cell.col} · ${loc.cell.header || 'no header'} / ${loc.cell.row_label || 'no row label'}`);
  const where = cellOnly ? 'The cited cell is marked in the table; its row was not found in the page text, so the text is not marked.'
    : hl === null ? 'The snippet was not found in this page text.'
    : hl.approx ? 'Approximate location (whitespace-insensitive match in the browser, not the verifier’s span).'
    : 'Marked at the verifier’s span.';
  const title = `${record.field}${record.group_label ? ` · ${record.group_label}` : ''}`;

  return (
    <div className="ev-context" onKeyDown={e => { if (e.key === 'Escape' && onClose) { e.stopPropagation(); onClose(); } }}>
      <div className="ev-context-head">
        <h3 ref={headRef} tabIndex={-1}>
          Cited page for {title} = <span className="mono">{record.value}</span>{record.unit ? ` ${record.unit}` : ''}
          {(urgent || record.flags.length > 0 || view.ready?.meta.compare_urgent) && (
            <span className="ev-compare-label"><OctagonAlert size={12} aria-hidden="true" /> {compareLabel(decided, urgent)}</span>
          )}
        </h3>
        {onClose && (
          <button type="button" className="btn btn-ghost" onClick={onClose} aria-label={`Close cited page for ${title}`}>
            Close
          </button>
        )}
      </div>
      <p className="ev-prov">{filename} · {parts.join(' · ')}</p>
      <p className="sr-only" role="status">
        {current ? `Page ${loc.page} loaded for ${title}. ${where}` : error ? '' : `Loading page ${loc.page}`}
        {view.ready && !view.failure?.broken ? ` ${imageAnnouncement(view.ready.meta, decided)}` : ''}
        {view.unavailable ? ` ${imageUnavailableText(urgent, decided)}` : ''}
      </p>
      <PageImage view={view} record={record} filename={filename} source={source} rebuild={rebuild} onRebuilt={view.reload} />
      {decide}
      <p className="ev-snippet"><span className="ev-label">Cited snippet</span> <span className="mono">{loc.snippet}</span></p>
      {error && <p className="ev-problem" role="alert">Could not load page {loc.page}: {error}</p>}
      {!current && !error && <p className="ev-empty">Loading page {loc.page}…</p>}
      {table && loc.cell && <CellGrid table={table} cell={loc.cell} literal={loc.literal} token={loc.token} />}
      {current && loc.kind === 'table' && !table && (
        <p className="ev-problem">Table {loc.cell ? loc.cell.table + 1 : '?'} is not in the parsed page.</p>
      )}
      {ctx && (
        <>
          <p className="ev-hint">{where}</p>
          <pre ref={boxRef} className="ev-pagetext" tabIndex={0} aria-label={`Text of page ${loc.page}`}>
            {hl ? <>{hl.before}<mark ref={markRef} className={hl.approx ? 'approx' : undefined}><Hit hit={hl.hit} literal={loc.literal} token={loc.token} start={hl.start} /></mark>{hl.after}</> : ctx.text}
          </pre>
        </>
      )}
    </div>
  );
}
