import { AlertTriangle, Check, CheckCircle, XCircle } from 'lucide-react';
import type { NcaSubject, QcCheck, QcIssue } from '../types';
import { shortHash } from './format';

/** Friendlier names for the backend's QC check strings (backend/app/tools/
 *  qc_tools.py). Anything not listed renders its raw check name. */
const QC_CHECK_LABEL: Record<string, string> = {
  'Sample size adequacy': 'Sample size',
  'Lambda_z points >= 3': 'λz points',
  'Lambda_z adj R^2 >= 0.80': 'λz adjusted R²',
  'AUC %extrap <= 20%': 'AUC extrapolation',
};

/** Checks whose remedy is the λz panel in the document. */
const LZ_CHECK = /Lambda_z|AUC %extrap/;

const plural = (n: number, word: string) => `${n} ${word}${n === 1 ? '' : 's'}`;
/** '20.0' → '20', '12.5' → '12.5'. */
const num = (v: string | number) => String(Number(v));
const pct1 = (v: number) => String(Math.round(v * 10) / 10);

/** 'subject 1' | 'subjects 1 and 4' | 'subjects 1, 4 and 7'. */
function subjectList(rows: NcaSubject[]): string {
  const ids = rows.map(r => String(r.subject));
  const list = ids.length > 1 ? `${ids.slice(0, -1).join(', ')} and ${ids[ids.length - 1]}` : ids[0];
  return `${ids.length === 1 ? 'subject' : 'subjects'} ${list}`;
}

/** The NCA rows `test` flags, or null when we hold no rows or our count
 *  disagrees with the backend's `n` — never name ids the QC tool did not flag. */
function flaggedRows(rows: NcaSubject[] | null | undefined, n: number,
                     test: (r: NcaSubject) => boolean): NcaSubject[] | null {
  if (!rows?.length) return null;
  const hit = rows.filter(test);
  return hit.length === n ? hit : null;
}

type DetailFormatter = (detail: string, check: string, rows?: NcaSubject[] | null) => string | null;

/** Plain-language rewrites of each check's `detail` string (backend/app/
 *  tools/qc_tools.py). Limits come from the payload text, flagged subjects
 *  from the NCA rows when they reproduce the backend's count. A formatter
 *  returns null when the string is not the shape it expects, and the row
 *  then shows the raw detail. */
const QC_CHECK_DETAIL: Record<string, DetailFormatter> = {
  'Sample size adequacy': d => {
    const m = /^(\d+) subjects? \(>= (\d+) recommended\)$/.exec(d);
    return m && `${plural(Number(m[1]), 'subject')}, at least ${m[2]} needed`;
  },
  'Missing data': d => {
    const m = /^([\d.]+)% missing \(<= ([\d.]+)%\)$/.exec(d);
    return m && `${num(m[1])} %, limit ${num(m[2])} %`;
  },
  'Lambda_z points >= 3': (d, _c, rows) => {
    const m = /^(\d+) subject\(s\) with < (\d+) points$/.exec(d);
    if (!m) return null;
    const n = Number(m[1]), min = Number(m[2]);
    if (n === 0) return `all subjects use ${min} or more`;
    const hit = flaggedRows(rows, n, r => (r.lambda_z_n_points ?? 0) < min);
    if (!hit) return `${plural(n, 'subject')} use${n === 1 ? 's' : ''} fewer than ${min}`;
    const each = hit.map(r => `subject ${r.subject} uses ${plural(r.lambda_z_n_points ?? 0, 'point')}`);
    return `${each.join('; ')}, at least ${min} needed`;
  },
  'Lambda_z adj R^2 >= 0.80': (d, check, rows) => {
    const m = /^(\d+) subject\(s\) below threshold$/.exec(d);
    const thr = />= ([\d.]+)$/.exec(check)?.[1];
    if (!m || !thr) return null;
    const n = Number(m[1]);
    if (n === 0) return `all subjects at ${thr} or better`;
    const hit = flaggedRows(rows, n, r => r.lambda_z_r2_adj != null && r.lambda_z_r2_adj < Number(thr));
    if (!hit) return `${plural(n, 'subject')} below ${thr}`;
    const each = hit.map(r => `subject ${r.subject} at ${(r.lambda_z_r2_adj ?? 0).toFixed(2)}`);
    return `${each.join('; ')}, limit ${thr}`;
  },
  'AUC %extrap <= 20%': (d, _c, rows) => {
    const m = /^(\d+) subject\(s\) > ([\d.]+)%$/.exec(d);
    if (!m) return null;
    const n = Number(m[1]), lim = num(m[2]);
    if (n === 0) return `all subjects within ${lim} %`;
    const hit = flaggedRows(rows, n, r => r.pct_AUC_extrap != null && r.pct_AUC_extrap > Number(lim));
    if (!hit) return `${plural(n, 'subject')} over ${lim} %`;
    const each = hit.map(r => `subject ${r.subject} at ${pct1(r.pct_AUC_extrap ?? 0)} %`);
    return `${each.join('; ')}, limit ${lim} %`;
  },
  'Parameter plausibility': (d, _c, rows) => {
    const m = /^(\d+) subject\(s\) with non-physiological CL\/V$/.exec(d);
    if (!m) return null;
    const n = Number(m[1]);
    if (n === 0) return 'every CL and V is above zero';
    const hit = flaggedRows(rows, n, r => (r.CL_F != null && r.CL_F <= 0) || (r.Vz_F != null && r.Vz_F <= 0));
    return hit
      ? `${subjectList(hit)} ${hit.length === 1 ? 'has' : 'have'} a CL or V at or below zero`
      : `${plural(n, 'subject')} with a CL or V at or below zero`;
  },
  'Tmax before terminal phase': (d, _c, rows) => {
    const m = /^(\d+) subject\(s\) with late Tmax$/.exec(d);
    if (!m) return null;
    const n = Number(m[1]);
    if (n === 0) return 'every subject peaks before the last sample';
    const hit = flaggedRows(rows, n, r => r.Tmax != null && r.Tlast != null && r.Tmax >= r.Tlast);
    return hit
      ? `${subjectList(hit)} peak${hit.length === 1 ? 's' : ''} at the last sample`
      : `${plural(n, 'subject')} with Tmax at the last sample`;
  },
};

/** Backend issue strings (qc_tools.py) → the checklist row they restate and
 *  a plain-language label. The capture is a Python list repr or a number. */
const QC_ISSUE: { re: RegExp; check: string; label: string; list: boolean }[] = [
  { re: /^only (\d+) subjects$/, check: 'Sample size adequacy', label: 'only $ subjects', list: false },
  { re: /^([\d.]+)% missing data$/, check: 'Missing data', label: '$ % missing data', list: false },
  { re: /^few lambda_z points: (\[.*\])$/, check: 'Lambda_z points >= 3', label: 'too few λz points', list: true },
  { re: /^poor terminal fit: (\[.*\])$/, check: 'Lambda_z adj R^2 >= 0.80', label: 'poor terminal fit', list: true },
  { re: /^high %extrap: (\[.*\])$/, check: 'AUC %extrap <= 20%', label: 'high % extrapolated', list: true },
  { re: /^implausible params: (\[.*\])$/, check: 'Parameter plausibility', label: 'implausible CL or V', list: true },
];

/** "[1, 'A']" → 'subject 1' | 'subjects 1 and A'. */
function reprSubjects(repr: string): string {
  const ids = repr.slice(1, -1).split(',').map(t => t.trim().replace(/^['"]|['"]$/g, '')).filter(Boolean);
  if (!ids.length) return '';
  const list = ids.length > 1 ? `${ids.slice(0, -1).join(', ')} and ${ids[ids.length - 1]}` : ids[0];
  return `${ids.length === 1 ? 'subject' : 'subjects'} ${list}`;
}

/** The checklist row an issue restates, or null when it maps to none. */
function issueCheck(iss: QcIssue): string | null {
  return QC_ISSUE.find(m => m.re.test(iss.issue))?.check ?? null;
}

/** 'Medium · subject 1: high % extrapolated'; unknown shapes keep their text. */
function issueText(iss: QcIssue): string {
  const sev = sentenceCase(iss.severity);
  for (const m of QC_ISSUE) {
    const hit = m.re.exec(iss.issue);
    if (!hit) continue;
    if (!m.list) return `${sev} · ${m.label.replace('$', num(hit[1]))}`;
    const who = reprSubjects(hit[1]);
    return who ? `${sev} · ${who}: ${m.label}` : `${sev} · ${m.label}`;
  }
  return `${sev} · ${iss.issue}`;
}

/** The check's detail as a sentence, or the raw detail when no formatter fits. */
function qcCheckDetail(c: QcCheck, rows?: NcaSubject[] | null): string {
  return QC_CHECK_DETAIL[c.check]?.(c.detail, c.check, rows) ?? c.detail;
}

function sentenceCase(v: string): string {
  return v ? v.charAt(0).toUpperCase() + v.slice(1).toLowerCase() : '';
}

function verdictTone(verdict: string): 'pass' | 'conditional' | 'fail' {
  if (verdict === 'PASS') return 'pass';
  if (verdict.includes('CONDITIONAL')) return 'conditional';
  return 'fail';
}

/** The QC agent's verdict, checklist and issues, as rows a reviewer can act
 *  on. Every row carries an icon AND an sr-only word, so colour is never the
 *  only signal. `seal` is the run_qc audit hash (passed in, never computed).
 *  `subjects` are the NCA rows the QC ran on, used only to name flagged ids.
 *  The review panel renders the full checklist; `compact` (the document's QC
 *  section, QcCard) renders one verdict line plus a 'Review in panel' link. */
export function QcEvidence({ verdict, checklist, issues, seal, subjects, onReviewLz, onReviewInPanel, compact }: {
  verdict: string;
  checklist: QcCheck[] | null;
  issues: QcIssue[] | null;
  seal?: string | null;
  subjects?: NcaSubject[] | null;
  onReviewLz?: () => void;
  onReviewInPanel?: () => void;
  compact?: boolean;
}) {
  const tone = verdictTone(verdict);
  const total = checklist?.length ?? 0;
  const passed = checklist?.filter(c => c.status === 'PASS').length ?? 0;
  const iconSize = compact ? 16 : 20;
  // Issues that restate a flagged checklist row add nothing; show the block
  // only when at least one issue has no failed row of its own.
  const flaggedChecks = new Set((checklist ?? []).filter(c => c.status !== 'PASS').map(c => c.check));
  const issuesCovered = (issues ?? []).every(iss => {
    const check = issueCheck(iss);
    return check != null && flaggedChecks.has(check);
  });
  const issueCount = issuesCovered ? 0 : issues?.length ?? 0;
  const flags = total - passed;

  // The document carries one verdict line; the checklist lives in the panel.
  if (compact) {
    return (
      <section className="qc-evidence compact qc-line" aria-label="QC summary">
        {tone === 'pass' && <CheckCircle size={iconSize} style={{ color: 'var(--green)' }} aria-hidden="true" />}
        {tone === 'conditional' && <AlertTriangle size={iconSize} style={{ color: 'var(--yellow)' }} aria-hidden="true" />}
        {tone === 'fail' && <XCircle size={iconSize} style={{ color: 'var(--red)' }} aria-hidden="true" />}
        <span className="qc-evidence-title">QC: {sentenceCase(verdict)}</span>
        {checklist && (
          <span className="qc-evidence-sub">
            · {passed} of {total}{flags > 0 && <> · {plural(flags, 'flag')}</>}
          </span>
        )}
        {onReviewInPanel && (
          <button type="button" className="link-btn" onClick={onReviewInPanel}>Review in panel</button>
        )}
      </section>
    );
  }

  return (
    <section className="qc-evidence" aria-label="QC evidence">
      <div className="qc-evidence-head">
        {tone === 'pass' && <CheckCircle size={iconSize} style={{ color: 'var(--green)' }} aria-hidden="true" />}
        {tone === 'conditional' && <AlertTriangle size={iconSize} style={{ color: 'var(--yellow)' }} aria-hidden="true" />}
        {tone === 'fail' && <XCircle size={iconSize} style={{ color: 'var(--red)' }} aria-hidden="true" />}
        <div className="qc-evidence-titles">
          <span className="qc-evidence-title">QC: {sentenceCase(verdict)}</span>
          <span className="qc-evidence-sub">
            {checklist && <>{passed} of {total} checks passed · </>}
            QC agent
            {seal && <> · sealed <code>{shortHash(seal)}</code></>}
          </span>
        </div>
      </div>

      {checklist && checklist.length > 0 && (
        <ul className="qc-list">
          {checklist.map((c, i) => {
            const cls = c.status === 'WARN' ? 'warn' : c.status === 'FAIL' ? 'fail' : '';
            const word = c.status === 'PASS' ? 'passed' : c.status === 'WARN' ? 'warning' : 'failed';
            const lz = LZ_CHECK.test(c.check) && c.status !== 'PASS' && onReviewLz;
            const detail = qcCheckDetail(c, subjects);
            return (
              <li key={i} className={cls}>
                {c.status === 'PASS' && <Check size={14} style={{ color: 'var(--green)' }} aria-hidden="true" />}
                {c.status === 'WARN' && <span className="badge-warn" aria-hidden="true">!</span>}
                {c.status === 'FAIL' && <XCircle size={14} style={{ color: 'var(--red)' }} aria-hidden="true" />}
                <span className="qc-row-text">
                  <span className="sr-only">{word} </span>
                  <span className="qc-row-name">{QC_CHECK_LABEL[c.check] ?? c.check}</span>
                  {detail && <span className="qc-row-detail"> · {detail}</span>}
                  {lz && (
                    <>
                      {' '}
                      <button type="button" className="link-btn" onClick={onReviewLz}>Review λz</button>
                    </>
                  )}
                </span>
              </li>
            );
          })}
        </ul>
      )}

      {issueCount > 0 && (
        <details className="qc-issues-details">
          <summary>{plural(issueCount, 'issue')}</summary>
          <ul className="qc-issues">
            {(issues ?? []).map((iss, i) => (
              <li key={i}>{issueText(iss)}</li>
            ))}
          </ul>
        </details>
      )}

        </section>
  );
}
