/** Shared shell types. Kept free of React so App.tsx and the shell
 *  components can import from here without a circular dependency. */

/** Every transcript result-card marker App.tsx renders. `CARD_META` (S4) is
 *  typed against this tuple, so a marker missing its metadata fails tsc. */
export const MARKERS = [
  '__SPAGHETTI__', '__NCA_TABLE__', '__NCA_LZ__', '__QC_CARD__', '__BE__', '__DP__',
  '__CLINPHARM__', '__STATS__', '__COMPARTMENTAL__', '__POPPK__', '__PENDING_TOOL__',
  '__PKMODEL__', '__VPC__', '__NLME__', '__PRIORCHECK__', '__SCM__', '__ENGINES__',
  '__FORECAST__', '__DIAG__', '__FOREST__', '__SIMEST__', '__BOOTSTRAP__', '__SIR__',
  '__PROFILE__', '__SWEEP__', '__CLINSIM__', '__EXPFOREST__', '__SPECIALPOP__',
  '__INDIVEXP__', '__PEDIATRIC__', '__SIM__', '__REVIEW__', '__REPORT__',
] as const;
export type Marker = typeof MARKERS[number];

/** Human decision at the review gate: null before any gate has been reached. */
export type Decision = 'pending' | 'approved' | 'rejected' | null;

/** Who the backend will record as the actor of a gate decision. Derived from
 *  /api/health `auth` plus the topbar token, never assumed: with auth open the
 *  backend ignores any token and seals the entry as `anonymous`. */
export type GateSigner =
  | 'token'      // auth required + token set → pseudonymous `token:<sha256[:16]>`
  | 'open'       // backend auth open → always `anonymous`; a typed token is ignored
  | 'anonymous'; // auth required (or unknown) and no token → `anonymous`

/** Verdict shown beside "Verify chain". `dim` is the hash-only (dev) mode:
 *  structure intact but nothing authenticates it, so it is never a pass. */
export type VerifyNote = { tone: 'ok' | 'dim' | 'bad'; text: string };

/** One row of a workflow's step tracker. */
export type StepView = { key: string; label: string; gate?: boolean };

export type WorkflowName = 'nca_full' | 'poppk_modeling' | 'poppk_full';

/** Below this width the review column is a fixed drawer (index.css: the
 *  `@media (max-width: 1180px)` block). App opens it when a gate arrives. */
export const REVIEW_DRAWER_MQ = '(max-width: 1180px)';

/** What the review panel shows once a gate has been decided (or a workflow
 *  finished without one). Built in App from live state + audit; nothing here
 *  is derived client-side beyond picking which card to show. */
export type Outcome =
  | { kind: 'report'; filename: string; onDocx: () => void; onCsv?: () => void; onCdisc?: () => void }
  | { kind: 'complete' }
  | { kind: 'running'; note: string }
  | { kind: 'rejected'; stepLabel: string; entryIndex: number | null; hint: string };
