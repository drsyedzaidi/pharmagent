import { useEffect, useRef } from 'react';
import type { CSSProperties, ReactNode, RefObject } from 'react';

const FOCUSABLE =
  'button:not([disabled]), [href], input:not([disabled]), select:not([disabled]), '
  + 'textarea:not([disabled]), [tabindex]:not([tabindex="-1"])';

/** Light-dismiss popover anchored to a button. The caller wraps anchor + popover
 *  in a `position: relative` element; the panel is absolutely positioned inside
 *  it. Escape and pointerdown outside (anchor + panel) call `onClose`; when the
 *  popover closes, focus returns to the anchor; a dialog moves focus into
 *  itself on open. No state lives here. */
export function Popover({
  open, anchorRef, onClose, role = 'dialog', align = 'left', placement = 'below',
  ariaLabel, className, id, children,
}: {
  open: boolean;
  anchorRef: RefObject<HTMLElement | null>;
  onClose: () => void;
  role?: 'dialog' | 'menu';
  align?: 'left' | 'right';
  placement?: 'below' | 'above';
  ariaLabel?: string;
  className?: string;
  id?: string;
  children: ReactNode;
}) {
  const panelRef = useRef<HTMLDivElement>(null);
  // Latest onClose lives in a ref so the listener effect depends on `open` only:
  // its cleanup (which returns focus to the anchor) then runs on close, not on
  // every parent re-render that hands us a new inline callback.
  const onCloseRef = useRef(onClose);
  useEffect(() => { onCloseRef.current = onClose; }, [onClose]);

  useEffect(() => {
    if (!open) return;
    const anchor = anchorRef.current;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') { e.stopPropagation(); onCloseRef.current(); }
    };
    const onPointer = (e: PointerEvent) => {
      const t = e.target as Node | null;
      if (!t) return;
      if (panelRef.current?.contains(t)) return;
      if (anchor?.contains(t)) return;
      onCloseRef.current();
    };
    document.addEventListener('keydown', onKey);
    document.addEventListener('pointerdown', onPointer);
    return () => {
      document.removeEventListener('keydown', onKey);
      document.removeEventListener('pointerdown', onPointer);
      // closing (or unmounting while open) hands focus back to the anchor
      if (anchor && document.contains(anchor)) anchor.focus();
    };
  }, [open, anchorRef]);

  // A dialog takes focus on open: its first enabled control, else the panel
  // itself (tabIndex -1). Menus manage their own item focus (ExportMenu).
  useEffect(() => {
    if (!open || role !== 'dialog') return;
    const panel = panelRef.current;
    if (!panel) return;
    const first = panel.querySelector<HTMLElement>(FOCUSABLE);
    (first ?? panel).focus();
  }, [open, role]);

  if (!open) return null;

  const style: CSSProperties = {
    position: 'absolute',
    zIndex: 20,
    minWidth: 200,
    background: 'var(--bg-card)',
    border: '1px solid var(--border)',
    borderRadius: 8,
    boxShadow: '0 8px 24px rgba(22, 49, 74, 0.14)',
    padding: 10,
    ...(align === 'right' ? { right: 0 } : { left: 0 }),
    ...(placement === 'above' ? { bottom: 'calc(100% + 6px)' } : { top: 'calc(100% + 6px)' }),
  };

  return (
    <div
      ref={panelRef}
      id={id}
      role={role}
      aria-label={ariaLabel}
      className={className}
      style={style}
      tabIndex={role === 'dialog' ? -1 : undefined}
    >
      {children}
    </div>
  );
}
