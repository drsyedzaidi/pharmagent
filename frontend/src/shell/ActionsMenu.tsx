import type { MouseEvent, ReactNode, RefObject } from 'react';
import { Popover } from './Popover';

/** The Actions popover above the composer. Its children are App's quick-action
 *  rows, moved verbatim, so every handler and predicate still lives in App.
 *
 *  Close rule (hybrid, deliberate): a click on any <button> inside the panel
 *  closes the menu AFTER the button's own handler has run, so a one-shot run
 *  ('Fit model', 'SAEM', a preset) leaves the new section visible. Buttons
 *  carrying `data-keep-open` (the panel toggles: Columns / roles, Flexplot,
 *  Skills, Calculators, Simulation-estimation…, Cancel) keep it open.
 *  Escape and an outside click close it too; Popover returns focus to the
 *  Actions button. Because all state stays in App, closing and reopening
 *  loses nothing (the simest confirm fields persist). */
export function ActionsMenu({ open, onClose, anchorRef, children }: {
  open: boolean;
  onClose: () => void;
  anchorRef: RefObject<HTMLButtonElement | null>;
  children: ReactNode;
}) {
  const onClick = (e: MouseEvent<HTMLDivElement>) => {
    const btn = (e.target as HTMLElement | null)?.closest('button');
    if (!btn || !e.currentTarget.contains(btn)) return;
    if (btn.hasAttribute('data-keep-open')) return;
    // bubble phase: runs after the button's own onClick
    onClose();
  };
  return (
    <Popover
      id="actions-pop"
      open={open}
      onClose={onClose}
      anchorRef={anchorRef}
      role="dialog"
      ariaLabel="Actions"
      placement="above"
      className="actions-menu"
    >
      <div className="actions-menu-body" onClick={onClick}>
        {children}
      </div>
    </Popover>
  );
}
