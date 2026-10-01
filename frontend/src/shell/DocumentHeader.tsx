import type { ReactNode } from 'react';

/** Document title block: h1, one-line subtitle built by App from live
 *  state (file, subjects, session start, sealed-entry count) and a right
 *  slot for the Export menu. */
export function DocumentHeader({ title, subtitle, right }: {
  title: string;
  subtitle: string;
  right?: ReactNode;
}) {
  return (
    <div className="doc-head">
      <div className="doc-head-text">
        <h1>{title}</h1>
        <p className="doc-subtitle">{subtitle}</p>
      </div>
      {right}
    </div>
  );
}
