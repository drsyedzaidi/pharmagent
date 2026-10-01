import type { ReactNode } from 'react';
import { agentLabel } from './agents';
import { shortHash } from './format';

/** One section of the analysis document: a heading, optional inline
 *  controls, and a quiet right-hand attribution line built ONLY from the
 *  parts that exist — agent, backend tool name, and the audit seal for that
 *  tool (the entry bound to the card's own snapshot, or the latest entry for
 *  a live card; never a client-side hash). `sealing` is shown only while a
 *  registered tool has no seal yet. Presentational:
 *  the card itself is passed as children, untouched. */
export function ResultSection({
  title, agent, tool, seal, sealing, maxWidth, id, controls, children,
}: {
  title: string;
  agent?: string;
  tool?: string;
  seal?: string | null;
  sealing?: boolean;
  maxWidth?: number;
  id?: string;
  controls?: ReactNode;
  children: ReactNode;
}) {
  const who = agentLabel(agent);
  const hasAttr = !!(who || tool || seal || sealing);
  return (
    <section id={id} className="section">
      <div className="section-head">
        <h2>{title}</h2>
        {controls}
        {hasAttr && (
          <span className="section-attr">
            {who}
            {tool && <>{who ? ' · ' : ''}<code>{tool}</code></>}
            {seal
              ? <>{' · sealed '}<code>{shortHash(seal)}</code></>
              : sealing ? ' · sealing…' : null}
          </span>
        )}
      </div>
      <div className="section-body" style={maxWidth ? { maxWidth } : undefined}>
        {children}
      </div>
    </section>
  );
}
