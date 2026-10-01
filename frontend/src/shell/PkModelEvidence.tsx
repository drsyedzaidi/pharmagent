import type { PkModelResults } from '../types';
import { agentLabel } from './agents';
import { shortHash } from './format';

type Row = { key: string; label: string; aic: number | null };

/** The ranking as rows: compare mode uses the backend's order (pk_fit ranks
 *  on total AIC); fit mode is the one model that was fitted. */
function rowsOf(r: PkModelResults): Row[] {
  if (r.mode === 'compare' && r.ranking) {
    return r.ranking.map(row => ({ key: row.model_key, label: row.label, aic: row.total_aic }));
  }
  return [{ key: r.model_key ?? 'fit', label: r.label ?? r.model_key ?? 'Model', aic: r.total_aic ?? null }];
}

/** Evidence for the fit_pk_model gate, sized for the review panel: a ranked
 *  two-column list (model · total AIC) with the winner marked by the word
 *  "best", and a jump to the full comparison in the document. The parameter
 *  tables stay in the document section. `seal` is the fit_pk_model audit
 *  hash (passed in, never computed). */
export function PkModelEvidence({ r, seal, onSeeComparison }: {
  r: PkModelResults;
  seal?: string | null;
  onSeeComparison: () => void;
}) {
  if (r.status !== 'ok') {
    return (
      <section className="pk-evidence" aria-label="Structural model evidence">
        <span className="pk-evidence-title">Structural models: not run</span>
        {r.message && <span className="pk-evidence-sub">{r.message}</span>}
      </section>
    );
  }
  const rows = rowsOf(r);
  const isCompare = r.mode === 'compare' && rows.length > 1;
  const bestKey = isCompare ? (r.best_model ?? rows.find(x => x.aic != null)?.key ?? null) : null;
  return (
    <section className="pk-evidence" aria-label="Structural model evidence">
      <div className="pk-evidence-titles">
        <span className="pk-evidence-title">
          {isCompare ? `Structural models: ${rows.length} compared` : 'Structural model fitted'}
        </span>
        <span className="pk-evidence-sub">
          {isCompare ? 'Ranked by total AIC · ' : ''}
          {agentLabel('modeler')}
          {seal && <> · sealed <code>{shortHash(seal)}</code></>}
        </span>
      </div>
      <ol className="pk-rank">
        {rows.map(row => {
          const best = row.key === bestKey;
          return (
            <li key={row.key} className={best ? 'best' : ''}>
              <span className="pk-rank-model">{row.label}</span>
              <span className="pk-rank-aic">
                {best && <span className="pk-rank-best">best</span>}
                {row.aic == null ? <span className="pk-rank-nofit">no fit</span> : row.aic.toFixed(1)}
              </span>
            </li>
          );
        })}
      </ol>
      <button type="button" className="link-btn" onClick={onSeeComparison}>
        {isCompare ? 'See comparison in the document' : 'See the fit in the document'}
      </button>
    </section>
  );
}
