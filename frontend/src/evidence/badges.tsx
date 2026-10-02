/** Small status vocabulary shared by the Screening and Records tabs and the chat cards.
 *  Colour is never the only signal: every badge pairs a lucide icon with a word. */
import { AlertTriangle, CheckCircle, CircleQuestionMark, Clock, OctagonAlert, XCircle } from 'lucide-react';
import { failWords } from './format';
import type { DecisionStatus, Inclusion, QcFlag, Verdict, VerifyFields } from './types';

const VERDICT: Record<Verdict, { cls: string; Icon: typeof CheckCircle }> = {
  include: { cls: 'approved', Icon: CheckCircle },
  exclude: { cls: 'rejected', Icon: XCircle },
  unsure: { cls: 'pending', Icon: CircleQuestionMark },
};

export function VerdictPill({ verdict }: { verdict: Verdict }) {
  const { cls, Icon } = VERDICT[verdict];
  return <span className={`pill ${cls}`}><Icon size={12} aria-hidden="true" /> {verdict}</span>;
}

/** The reviewer's inclusion decision for a publication (what extraction acts on). */
export function InclusionPill({ inclusion }: { inclusion: Inclusion }) {
  const Icon = inclusion === 'include' ? CheckCircle : XCircle;
  return (
    <span className={`pill ${inclusion === 'include' ? 'approved' : 'rejected'}`}>
      <Icon size={12} aria-hidden="true" /> {inclusion === 'include' ? 'included' : 'excluded'}
    </span>
  );
}

const STATUS: Record<DecisionStatus, { cls: string; Icon: typeof CheckCircle }> = {
  proposed: { cls: 'ev-neutral', Icon: Clock },
  approved: { cls: 'approved', Icon: CheckCircle },
  rejected: { cls: 'rejected', Icon: XCircle },
};

export function StatusPill({ status }: { status: DecisionStatus }) {
  const { cls, Icon } = STATUS[status];
  return <span className={`pill ${cls}`}><Icon size={12} aria-hidden="true" /> {status}</span>;
}

export function VerifyBadge({ rec }: { rec: VerifyFields }) {
  if (rec.verify_status === 'pass') {
    return <span className="ev-verify ok"><CheckCircle size={13} aria-hidden="true" /> verified</span>;
  }
  return (
    <span className="ev-verify bad" title={rec.verify_reason}>
      <XCircle size={13} aria-hidden="true" /> not found: {failWords(rec)}
    </span>
  );
}

export function FlagChip({ flag }: { flag: QcFlag }) {
  const Icon = flag.severity === 'error' ? OctagonAlert : AlertTriangle;
  const origin = flag.origin === 'llm' ? ' (model)' : '';
  return (
    <span className={`ev-flag ${flag.severity}`} title={`${flag.severity}: ${flag.message}${origin}`}>
      <Icon size={12} aria-hidden="true" />
      <span className="sr-only">{flag.severity}: </span>{flag.code.replace(/_/g, ' ')}{origin}
    </span>
  );
}

/** A screening criterion's assessment: met / not met / not assessed. */
export function MetWord({ met }: { met: boolean | null }) {
  if (met === true) return <span className="ev-met ok"><CheckCircle size={12} aria-hidden="true" /> met</span>;
  if (met === false) return <span className="ev-met bad"><XCircle size={12} aria-hidden="true" /> not met</span>;
  return <span className="ev-met dim"><CircleQuestionMark size={12} aria-hidden="true" /> not assessed</span>;
}
