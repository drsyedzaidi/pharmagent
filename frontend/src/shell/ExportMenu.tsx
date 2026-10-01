import { useEffect, useRef, useState } from 'react';
import type { KeyboardEvent } from 'react';
import { ChevronDown } from 'lucide-react';
import { Popover } from './Popover';

export type ExportItem = { id: string; label: string; title?: string; onClick: () => void };

/** 'Export ▾' menu in the document header. Items come from App exactly as the
 *  old export chips did; the menu only adds keyboard semantics: ArrowDown /
 *  ArrowUp / Home / End move focus between items, Escape or an outside click
 *  closes (focus returns to the button via Popover), and an item click runs
 *  its handler then closes. */
export function ExportMenu({ disabled, items }: { disabled: boolean; items: ExportItem[] }) {
  const btnRef = useRef<HTMLButtonElement>(null);
  const listRef = useRef<HTMLDivElement>(null);
  const [open, setOpen] = useState(false);

  // First item takes focus when the menu opens (focus only; no state).
  useEffect(() => {
    if (!open) return;
    const first = listRef.current?.querySelector<HTMLButtonElement>('[role="menuitem"]');
    first?.focus();
  }, [open]);

  const onKeyDown = (e: KeyboardEvent<HTMLDivElement>) => {
    const nodes = Array.from(
      listRef.current?.querySelectorAll<HTMLButtonElement>('[role="menuitem"]') ?? [],
    );
    if (nodes.length === 0) return;
    const cur = nodes.findIndex(n => n === document.activeElement);
    let next = -1;
    if (e.key === 'ArrowDown') next = cur < 0 ? 0 : (cur + 1) % nodes.length;
    else if (e.key === 'ArrowUp') next = cur < 0 ? nodes.length - 1 : (cur - 1 + nodes.length) % nodes.length;
    else if (e.key === 'Home') next = 0;
    else if (e.key === 'End') next = nodes.length - 1;
    if (next < 0) return;
    e.preventDefault();
    nodes[next].focus();
  };

  return (
    <div style={{ position: 'relative', flexShrink: 0 }}>
      <button
        ref={btnRef}
        type="button"
        className="btn btn-ghost export-btn"
        aria-haspopup="menu"
        aria-expanded={open}
        disabled={disabled}
        onClick={() => setOpen(o => !o)}
      >
        Export <ChevronDown size={12} aria-hidden="true" />
      </button>
      <Popover open={open} anchorRef={btnRef} onClose={() => setOpen(false)} role="menu" align="right" ariaLabel="Export">
        <div ref={listRef} className="export-menu" onKeyDown={onKeyDown}>
          {items.map(it => (
            <button
              key={it.id}
              type="button"
              role="menuitem"
              className="export-item"
              title={it.title}
              onClick={() => { it.onClick(); setOpen(false); }}
            >
              {it.label}
            </button>
          ))}
        </div>
      </Popover>
    </div>
  );
}
