import { useEffect, useRef, useState } from 'react';
import { evidencePage } from './api';
import { errorMessage } from './format';
import { contextText, highlight } from './highlight';
import type { PageText, ParsedTable, RecordRow, TableCell } from './types';

/** The marked snippet, with the verified literal (when present in it) in bold. */
function Hit({ hit, literal }: { hit: string; literal: string | null }) {
  const at = literal ? hit.indexOf(literal) : -1;
  if (!literal || at < 0) return <>{hit}</>;
  return <>{hit.slice(0, at)}<strong>{literal}</strong>{hit.slice(at + literal.length)}</>;
}

/** The parsed table a table locator cites, with the cited cell marked. */
function CellGrid({ table, cell, literal }: { table: ParsedTable; cell: TableCell; literal: string | null }) {
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
                    {hit ? <mark><Hit hit={c} literal={literal} /></mark> : c}
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
 *  at the verifier's raw span, and for table locators the parsed grid with the cell marked.
 *  Page text is untrusted document content and is rendered as text nodes only. */
export function RecordContext({ sessionId, record, filename }: {
  sessionId: string;
  record: RecordRow;
  filename: string;
}) {
  const loc = record.locator;
  const [page, setPage] = useState<PageText | null>(null);
  const [error, setError] = useState('');
  const boxRef = useRef<HTMLPreElement>(null);
  const markRef = useRef<HTMLElement>(null);

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

  const current = page && page.source_id === record.source_id && page.page === loc.page ? page : null;
  const ctx = current ? contextText(current, loc) : null;
  const hl = ctx ? highlight(ctx.text, loc.span, loc.snippet) : null;
  const table = current && loc.kind === 'table' && loc.cell
    ? current.tables.find(t => t.index === loc.cell?.table) ?? null : null;

  // Scroll the mark into the middle third of the text box only (never the page; no animation).
  useEffect(() => {
    const box = boxRef.current, mark = markRef.current;
    if (box && mark) box.scrollTop = Math.max(0, mark.offsetTop - box.clientHeight / 3);
  }, [hl?.before, record.id]);

  const parts = [
    `page ${loc.page}${current ? ` of ${current.pages}` : ''}`,
    loc.kind,
    loc.literal !== null ? `literal ${loc.literal}` : 'no literal',
    `decimal '${loc.decimal_style}'`,
  ];
  if (ctx?.column !== null && ctx?.column !== undefined) parts.push(`column ${ctx.column + 1}`);
  if (loc.cell) parts.push(`cell r${loc.cell.row} c${loc.cell.col} · ${loc.cell.header || 'no header'} / ${loc.cell.row_label || 'no row label'}`);

  return (
    <div className="ev-context" aria-live="polite">
      <div className="ev-context-head">
        <h3>{record.field}{record.group_label ? ` · ${record.group_label}` : ''} = <span className="mono">{record.value}</span>{record.unit ? ` ${record.unit}` : ''}</h3>
        <p className="ev-prov">{filename} · {parts.join(' · ')}</p>
      </div>
      <p className="ev-snippet"><span className="ev-label">Cited snippet</span> <span className="mono">{loc.snippet}</span></p>
      {error && <p className="ev-problem" role="alert">Could not load page {loc.page}: {error}</p>}
      {!current && !error && <p className="ev-empty">Loading page {loc.page}…</p>}
      {table && loc.cell && <CellGrid table={table} cell={loc.cell} literal={loc.literal} />}
      {current && loc.kind === 'table' && !table && (
        <p className="ev-problem">Table {loc.cell ? loc.cell.table + 1 : '?'} is not in the parsed page.</p>
      )}
      {ctx && (
        <>
          <p className="ev-hint">
            {hl === null ? 'The snippet was not found in this page text.'
              : hl.approx ? 'Approximate location (whitespace-insensitive match in the browser, not the verifier’s span).'
              : 'Marked at the verifier’s span.'}
          </p>
          <pre ref={boxRef} className="ev-pagetext" tabIndex={0} aria-label={`Text of page ${loc.page}`}>
            {hl ? <>{hl.before}<mark ref={markRef} className={hl.approx ? 'approx' : undefined}><Hit hit={hl.hit} literal={loc.literal} /></mark>{hl.after}</> : ctx.text}
          </pre>
        </>
      )}
    </div>
  );
}
